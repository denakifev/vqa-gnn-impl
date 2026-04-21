"""
Tests for scripts/eval_vcr_qar.py — the end-to-end Q→AR joint evaluator.

These tests fabricate per-sample prediction files on disk in the exact
format that `src.trainer.inferencer.Inferencer` now writes, then run the
evaluator on them. They cover:

    - happy path: all samples joinable, reported numbers match by hand.
    - sample_id disagreement fails loudly by default.
    - --allow-missing falls back to the intersection.
    - missing `sample_id` in saved files is surfaced as a stderr warning
      and those files are skipped.
    - JSON output round-trips through the CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_vcr_qar  # noqa: E402


def _write_pred(dir_path: Path, idx: int, sample_id: str | None, pred: int, label: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "pred_label": torch.tensor(pred, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.long),
    }
    if sample_id is not None:
        payload["sample_id"] = sample_id
    torch.save(payload, dir_path / f"output_{idx}.pth")


class TestLoadPredictions:
    def test_loads_and_keys_by_sample_id(self, tmp_path):
        dir_ = tmp_path / "qa"
        _write_pred(dir_, 0, "annot-1", pred=2, label=2)
        _write_pred(dir_, 1, "annot-2", pred=0, label=1)

        preds = eval_vcr_qar.load_predictions(dir_)
        assert set(preds.keys()) == {"annot-1", "annot-2"}
        assert preds["annot-1"].pred_label == 2
        assert preds["annot-2"].label == 1

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eval_vcr_qar.load_predictions(tmp_path / "nope")

    def test_missing_sample_id_is_skipped_with_warning(self, tmp_path, capsys):
        dir_ = tmp_path / "qa"
        _write_pred(dir_, 0, "annot-1", pred=1, label=1)
        _write_pred(dir_, 1, None, pred=0, label=0)  # legacy file

        preds = eval_vcr_qar.load_predictions(dir_)
        assert set(preds.keys()) == {"annot-1"}
        err = capsys.readouterr().err
        assert "Skipped 1" in err

    def test_duplicate_sample_id_raises(self, tmp_path):
        dir_ = tmp_path / "qa"
        _write_pred(dir_, 0, "dup", pred=0, label=0)
        _write_pred(dir_, 1, "dup", pred=1, label=1)
        with pytest.raises(ValueError, match="Duplicate sample_id"):
            eval_vcr_qar.load_predictions(dir_)


class TestComputeJointReport:
    def _mk(self, pred: int, label: int, sid: str = "x") -> eval_vcr_qar.SamplePrediction:
        return eval_vcr_qar.SamplePrediction(sample_id=sid, pred_label=pred, label=label)

    def test_all_correct(self):
        qa = {sid: self._mk(1, 1, sid) for sid in ["a", "b", "c"]}
        qar = {sid: self._mk(2, 2, sid) for sid in ["a", "b", "c"]}
        r = eval_vcr_qar.compute_joint_report(qa, qar, allow_missing=False)
        assert r.qa_accuracy == 1.0
        assert r.qar_accuracy == 1.0
        assert r.qar_joint_accuracy == 1.0
        assert r.num_joined == 3

    def test_joint_requires_both_correct(self):
        qa = {
            "a": self._mk(1, 1, "a"),  # qa ok
            "b": self._mk(0, 1, "b"),  # qa wrong
        }
        qar = {
            "a": self._mk(0, 1, "a"),  # qar wrong
            "b": self._mk(2, 2, "b"),  # qar ok
        }
        r = eval_vcr_qar.compute_joint_report(qa, qar, allow_missing=False)
        assert r.qa_accuracy == 0.5
        assert r.qar_accuracy == 0.5
        # Neither sample has both tasks correct
        assert r.qar_joint_accuracy == 0.0

    def test_disagreement_fails_by_default(self):
        qa = {"a": self._mk(1, 1, "a")}
        qar = {"b": self._mk(1, 1, "b")}
        with pytest.raises(ValueError, match="Prediction sets disagree"):
            eval_vcr_qar.compute_joint_report(qa, qar, allow_missing=False)

    def test_allow_missing_uses_intersection(self):
        qa = {
            "a": self._mk(1, 1, "a"),
            "b": self._mk(0, 0, "b"),
        }
        qar = {
            "a": self._mk(2, 2, "a"),
            "c": self._mk(0, 1, "c"),
        }
        r = eval_vcr_qar.compute_joint_report(qa, qar, allow_missing=True)
        assert r.num_joined == 1
        assert r.num_qa_only == 1
        assert r.num_qar_only == 1
        assert "b" in r.missing_only_in_qa
        assert "c" in r.missing_only_in_qar

    def test_empty_intersection_raises(self):
        qa = {"a": self._mk(1, 1, "a")}
        qar = {"b": self._mk(1, 1, "b")}
        with pytest.raises(ValueError, match="No sample_ids"):
            eval_vcr_qar.compute_joint_report(qa, qar, allow_missing=True)


class TestCLI:
    def test_end_to_end_writes_json(self, tmp_path, capsys):
        qa_dir = tmp_path / "qa"
        qar_dir = tmp_path / "qar"
        _write_pred(qa_dir, 0, "a", pred=1, label=1)
        _write_pred(qa_dir, 1, "b", pred=0, label=1)
        _write_pred(qar_dir, 0, "a", pred=2, label=2)
        _write_pred(qar_dir, 1, "b", pred=1, label=1)

        out_path = tmp_path / "report.json"
        rc = eval_vcr_qar.main([
            "--qa-dir", str(qa_dir),
            "--qar-dir", str(qar_dir),
            "--output", str(out_path),
        ])
        assert rc == 0
        payload = json.loads(out_path.read_text())
        assert payload["num_joined"] == 2
        # Sample 'a' is correct on both → qa_acc=0.5, qar_acc=1.0, joint=0.5
        assert payload["qa_accuracy"] == pytest.approx(0.5)
        assert payload["qar_accuracy"] == pytest.approx(1.0)
        assert payload["qar_joint_accuracy"] == pytest.approx(0.5)

    def test_cli_fails_on_disagreement(self, tmp_path):
        qa_dir = tmp_path / "qa"
        qar_dir = tmp_path / "qar"
        _write_pred(qa_dir, 0, "a", pred=1, label=1)
        _write_pred(qar_dir, 0, "b", pred=1, label=1)

        with pytest.raises(ValueError, match="Prediction sets disagree"):
            eval_vcr_qar.main([
                "--qa-dir", str(qa_dir),
                "--qar-dir", str(qar_dir),
            ])
