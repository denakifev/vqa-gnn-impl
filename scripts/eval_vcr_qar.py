"""
End-to-end VCR Q→AR joint accuracy evaluator.

Usage:
    python scripts/eval_vcr_qar.py \\
        --qa-dir  saved/vcr_qa_preds/val \\
        --qar-dir saved/vcr_qar_preds/val \\
        [--output report.json] [--allow-missing]

Inputs are directories of per-sample `.pth` files produced by
`src.trainer.inferencer.Inferencer`. Each file stores a dict with
`pred_label`, `label`, and (starting with the paper-aligned baseline)
`sample_id` — the VCR `annot_id`.

The evaluator joins predictions by `sample_id`, computes:

    Q→A accuracy:  pred_qa  == label_qa        over joined set
    QA→R accuracy: pred_qar == label_qar       over joined set
    Q→AR joint:    both correct                over joined set

and prints a short report. Samples present in only one side are listed
under `missing_only_in_qa` / `missing_only_in_qar`. By default the
evaluator fails when the two sides do not agree on the set of
`sample_id`s; pass `--allow-missing` to evaluate only on the
intersection.

This script is pure post-hoc: it does not load any model, it does not
access the dataset, and it does not require the HF tokenizer. It is
therefore safe to run on a CPU-only machine after training on GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class SamplePrediction:
    sample_id: str
    pred_label: int
    label: int


@dataclass
class JointReport:
    num_joined: int
    num_qa_only: int
    num_qar_only: int
    qa_accuracy: float
    qar_accuracy: float
    qar_joint_accuracy: float
    missing_only_in_qa: list[str]
    missing_only_in_qar: list[str]

    def to_dict(self) -> dict:
        return {
            "num_joined": self.num_joined,
            "num_qa_only": self.num_qa_only,
            "num_qar_only": self.num_qar_only,
            "qa_accuracy": self.qa_accuracy,
            "qar_accuracy": self.qar_accuracy,
            "qar_joint_accuracy": self.qar_joint_accuracy,
            "missing_only_in_qa": self.missing_only_in_qa,
            "missing_only_in_qar": self.missing_only_in_qar,
        }


def _to_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def load_predictions(pred_dir: Path) -> dict[str, SamplePrediction]:
    """
    Load every `output_*.pth` under `pred_dir` and key them by `sample_id`.

    Files without a `sample_id` entry predate the paper-aligned baseline
    and are skipped with a warning printed to stderr; they cannot be
    joined reliably across runs.
    """
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

    preds: dict[str, SamplePrediction] = {}
    skipped_no_id = 0
    for path in sorted(pred_dir.glob("output_*.pth")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = payload.get("sample_id")
        if sample_id is None:
            skipped_no_id += 1
            continue
        sample_id = str(sample_id)
        if sample_id in preds:
            raise ValueError(
                f"Duplicate sample_id '{sample_id}' in {pred_dir}. "
                "Re-run inference so every sample appears at most once, "
                "or delete stale predictions before re-running."
            )
        preds[sample_id] = SamplePrediction(
            sample_id=sample_id,
            pred_label=_to_int(payload["pred_label"]),
            label=_to_int(payload["label"]),
        )
    if skipped_no_id:
        print(
            f"[warn] Skipped {skipped_no_id} prediction files under {pred_dir} "
            "that do not carry `sample_id`. Re-run inference with the "
            "current inferencer to produce joinable outputs.",
            file=sys.stderr,
        )
    return preds


def compute_joint_report(
    qa_preds: dict[str, SamplePrediction],
    qar_preds: dict[str, SamplePrediction],
    allow_missing: bool,
) -> JointReport:
    qa_ids = set(qa_preds)
    qar_ids = set(qar_preds)
    joined_ids = sorted(qa_ids & qar_ids)
    only_qa = sorted(qa_ids - qar_ids)
    only_qar = sorted(qar_ids - qa_ids)

    if (only_qa or only_qar) and not allow_missing:
        raise ValueError(
            f"Prediction sets disagree: {len(only_qa)} sample_ids appear only in "
            f"Q→A, {len(only_qar)} appear only in QA→R. Re-run both "
            "partitions against the same dataset, or pass --allow-missing "
            "to evaluate on the intersection."
        )
    if not joined_ids:
        raise ValueError(
            "No sample_ids are present in both prediction directories. "
            "Check that both runs used the same validation split."
        )

    qa_correct = 0
    qar_correct = 0
    joint_correct = 0
    for sid in joined_ids:
        qa = qa_preds[sid]
        qar = qar_preds[sid]
        qa_ok = qa.pred_label == qa.label
        qar_ok = qar.pred_label == qar.label
        qa_correct += int(qa_ok)
        qar_correct += int(qar_ok)
        joint_correct += int(qa_ok and qar_ok)

    n = len(joined_ids)
    return JointReport(
        num_joined=n,
        num_qa_only=len(only_qa),
        num_qar_only=len(only_qar),
        qa_accuracy=qa_correct / n,
        qar_accuracy=qar_correct / n,
        qar_joint_accuracy=joint_correct / n,
        missing_only_in_qa=only_qa,
        missing_only_in_qar=only_qar,
    )


def format_report(report: JointReport) -> str:
    lines = [
        "VCR Q→AR joint evaluation",
        "-------------------------",
        f"Joined samples        : {report.num_joined}",
        f"Only in Q→A           : {report.num_qa_only}",
        f"Only in QA→R          : {report.num_qar_only}",
        f"Q→A accuracy          : {report.qa_accuracy:.4f}",
        f"QA→R accuracy         : {report.qar_accuracy:.4f}",
        f"Q→AR joint accuracy   : {report.qar_joint_accuracy:.4f}",
    ]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", required=True, type=Path)
    parser.add_argument("--qar-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Evaluate on the intersection even when the two prediction "
            "sets disagree on sample_ids."
        ),
    )
    args = parser.parse_args(argv)

    qa_preds = load_predictions(args.qa_dir)
    qar_preds = load_predictions(args.qar_dir)
    report = compute_joint_report(qa_preds, qar_preds, allow_missing=args.allow_missing)

    print(format_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nWrote JSON report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
