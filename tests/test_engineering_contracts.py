"""
Engineering-contract tests for VQA-GNN (arXiv:2205.11501).

Covers the inter-component contracts that the rest of the repository depends on
for the GQA runtime paths:

  1. GQA data contract: d_kg mismatch fails fast with a clear error.
  2. GQA data contract: missing graph_edge_types fails fast by default,
     but can be explicitly opted out via `expect_graph_edge_types=False`.
  3. GQA data contract: missing required HDF5 datasets fails fast.
  4. Hydra inference config exposes `skip_model_load` as a structured key, so
     bare `inferencer.skip_model_load=True` works without the `+` prefix.
  5. Backward-compat imports remain available from src.model.

These tests must not rely on any real GQA artifacts under `data/`.
They synthesize tiny HDF5 fixtures in a tmp dir.

Run with:
    .venv/bin/python -m pytest tests/test_engineering_contracts.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures: minimal GQA-style HDF5 artifacts
# ---------------------------------------------------------------------------


def _make_answer_vocab(path: Path) -> None:
    path.write_text(json.dumps({
        "answer_to_idx": {"yes": 0, "no": 1},
        "idx_to_answer": ["yes", "no"],
        "metadata": {},
    }))


def _make_gqa_questions(path: Path, num_samples: int = 2) -> list[str]:
    payload = {}
    qids = []
    for i in range(num_samples):
        qid = f"q{i:04d}"
        qids.append(qid)
        payload[qid] = {
            "question": "is the sky blue?",
            "imageId": f"img{i}",
            "answer": "yes",
        }
    path.write_text(json.dumps(payload))
    return qids


def _write_gqa_fixture(
    tmp_path: Path,
    d_kg: int,
    *,
    include_edge_types: bool,
    include_attrs: bool = False,
    num_visual_nodes: int = 4,
    max_kg_nodes: int = 3,
    d_visual: int = 8,
    num_samples: int = 2,
) -> tuple[Path, Path]:
    """Build a minimal GQA directory layout under tmp_path. Returns (data_dir, vocab_path)."""
    import h5py

    data_dir = tmp_path / "gqa"
    (data_dir / "questions").mkdir(parents=True)
    (data_dir / "visual_features").mkdir(parents=True)
    (data_dir / "knowledge_graphs").mkdir(parents=True)

    vocab_path = data_dir / "gqa_answer_vocab.json"
    _make_answer_vocab(vocab_path)
    qids = _make_gqa_questions(
        data_dir / "questions" / "train_balanced_questions.json",
        num_samples=num_samples,
    )

    n_total = num_visual_nodes + 1 + max_kg_nodes
    with h5py.File(data_dir / "visual_features" / "train_features.h5", "w") as f:
        if include_attrs:
            f.attrs["num_boxes"] = num_visual_nodes
            f.attrs["feature_dim"] = d_visual
        for i in range(len(qids)):
            f.create_dataset(f"img{i}", data=np.zeros((num_visual_nodes, d_visual), dtype=np.float32))

    with h5py.File(data_dir / "knowledge_graphs" / "train_graphs.h5", "w") as f:
        if include_attrs:
            f.attrs["d_kg"] = d_kg
            f.attrs["num_visual_nodes"] = num_visual_nodes
            f.attrs["max_kg_nodes"] = max_kg_nodes
        for qid in qids:
            grp = f.create_group(qid)
            grp.create_dataset("node_features", data=np.zeros((2, d_kg), dtype=np.float32))
            grp.create_dataset("adj_matrix", data=np.zeros((n_total, n_total), dtype=np.float32))
            grp.create_dataset("node_types", data=np.zeros((n_total,), dtype=np.int32))
            if include_edge_types:
                grp.create_dataset(
                    "graph_edge_types",
                    data=np.zeros((n_total, n_total), dtype=np.int64),
                )
    return data_dir, vocab_path


# ---------------------------------------------------------------------------
# 1–3. GQA data contract
# ---------------------------------------------------------------------------


class TestGQADataContract:
    def _load_local_tokenizer(self, monkeypatch):
        """Avoid downloading `roberta-large` during tests."""
        from unittest.mock import MagicMock

        class _FakeTokenizer:
            def __call__(self, *args, **kwargs):
                L = kwargs.get("max_length", 8)
                import torch

                return {
                    "input_ids": torch.zeros(1, L, dtype=torch.long),
                    "attention_mask": torch.ones(1, L, dtype=torch.long),
                }

        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = _FakeTokenizer()
        monkeypatch.setattr("transformers.AutoTokenizer", mock_auto)

    def test_dkg_mismatch_fails_fast_with_diagnostic(self, tmp_path, monkeypatch):
        """Stored d_kg=300 but config expects d_kg=600 → clear ValueError on first __getitem__."""
        from src.datasets import GQADataset

        self._load_local_tokenizer(monkeypatch)
        data_dir, vocab_path = _write_gqa_fixture(tmp_path, d_kg=300, include_edge_types=True)

        ds = GQADataset(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(vocab_path),
            d_visual=8,
            d_kg=600,
            num_visual_nodes=4,
            max_kg_nodes=3,
            max_question_len=8,
        )
        with pytest.raises(ValueError, match="GQA data contract") as exc:
            _ = ds[0]
        msg = str(exc.value)
        assert "d_kg=300" in msg
        assert "d_kg=600" in msg
        assert "train_graphs.h5" in msg
        # Remediation hint mentions both fix options
        assert "prepare_gqa_data.py" in msg

    def test_missing_edge_types_fails_fast_by_default(self, tmp_path, monkeypatch):
        """Legacy artifact without graph_edge_types → clear ValueError by default."""
        from src.datasets import GQADataset

        self._load_local_tokenizer(monkeypatch)
        data_dir, vocab_path = _write_gqa_fixture(tmp_path, d_kg=300, include_edge_types=False)

        ds = GQADataset(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(vocab_path),
            d_visual=8,
            d_kg=300,
            num_visual_nodes=4,
            max_kg_nodes=3,
            max_question_len=8,
        )
        with pytest.raises(ValueError, match="graph_edge_types") as exc:
            _ = ds[0]
        msg = str(exc.value)
        assert "expect_graph_edge_types" in msg

    def test_missing_edge_types_allowed_with_explicit_opt_in(self, tmp_path, monkeypatch, caplog):
        """Explicit `expect_graph_edge_types=False` accepts older artifacts with a warning."""
        from src.datasets import GQADataset

        self._load_local_tokenizer(monkeypatch)
        data_dir, vocab_path = _write_gqa_fixture(tmp_path, d_kg=300, include_edge_types=False)

        ds = GQADataset(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(vocab_path),
            d_visual=8,
            d_kg=300,
            num_visual_nodes=4,
            max_kg_nodes=3,
            max_question_len=8,
            expect_graph_edge_types=False,
        )
        with caplog.at_level("WARNING"):
            item = ds[0]
        # Keys still conform to contract; edge types are zero-filled
        assert item["graph_edge_types"].sum().item() == 0
        # Warning explicitly marks run as not paper-faithful
        assert any("not paper-faithful" in rec.message for rec in caplog.records)

    def test_strict_schema_false_disables_sample_probe(self, tmp_path, monkeypatch):
        """`strict_graph_schema=False` skips the first-open probe.

        The underlying runtime can still fail later, but the fast path is off.
        This exists for debugging, not for hiding mismatches.
        """
        from src.datasets import GQADataset

        self._load_local_tokenizer(monkeypatch)
        data_dir, vocab_path = _write_gqa_fixture(tmp_path, d_kg=300, include_edge_types=True)

        # d_kg mismatch that *would* be caught by the probe is now deferred.
        ds = GQADataset(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(vocab_path),
            d_visual=8,
            d_kg=600,
            num_visual_nodes=4,
            max_kg_nodes=3,
            max_question_len=8,
            strict_graph_schema=False,
        )
        # Probe does not run — but the underlying tensor copy still fails later.
        with pytest.raises((ValueError, RuntimeError)):
            _ = ds[0]

    def test_missing_required_dataset_fails_fast(self, tmp_path, monkeypatch):
        """Graph group missing `adj_matrix` → clear ValueError on first-open probe."""
        import h5py

        from src.datasets import GQADataset

        self._load_local_tokenizer(monkeypatch)
        data_dir, vocab_path = _write_gqa_fixture(tmp_path, d_kg=300, include_edge_types=True)

        # Corrupt the graph file: drop `adj_matrix`
        graph_file = data_dir / "knowledge_graphs" / "train_graphs.h5"
        with h5py.File(graph_file, "a") as f:
            for qid in f.keys():
                del f[qid]["adj_matrix"]

        ds = GQADataset(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(vocab_path),
            d_visual=8,
            d_kg=300,
            num_visual_nodes=4,
            max_kg_nodes=3,
            max_question_len=8,
        )
        with pytest.raises(ValueError, match="adj_matrix"):
            _ = ds[0]

    def test_attrs_based_mismatch_fails_fast(self, tmp_path, monkeypatch):
        """New-style artifact with declared attrs.d_kg=300 vs config d_kg=600."""
        from src.datasets import GQADataset

        self._load_local_tokenizer(monkeypatch)
        data_dir, vocab_path = _write_gqa_fixture(
            tmp_path, d_kg=300, include_edge_types=True, include_attrs=True,
        )

        ds = GQADataset(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(vocab_path),
            d_visual=8,
            d_kg=600,
            num_visual_nodes=4,
            max_kg_nodes=3,
            max_question_len=8,
        )
        with pytest.raises(ValueError, match="d_kg=300"):
            _ = ds[0]

    def test_random_subset_is_reproducible_for_a_fixed_seed(self, tmp_path, monkeypatch):
        from src.datasets import GQADataset

        self._load_local_tokenizer(monkeypatch)
        data_dir, vocab_path = _write_gqa_fixture(
            tmp_path,
            d_kg=300,
            include_edge_types=True,
            num_samples=8,
        )

        common_kwargs = dict(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(vocab_path),
            d_visual=8,
            d_kg=300,
            num_visual_nodes=4,
            max_kg_nodes=3,
            max_question_len=8,
            limit=4,
            shuffle_index=True,
        )
        ds_a = GQADataset(**common_kwargs, shuffle_seed=123)
        ds_b = GQADataset(**common_kwargs, shuffle_seed=123)
        ds_c = GQADataset(**common_kwargs, shuffle_seed=124)

        subset_a = [row["question_id"] for row in ds_a._index]
        subset_b = [row["question_id"] for row in ds_b._index]
        subset_c = [row["question_id"] for row in ds_c._index]

        assert subset_a == subset_b
        assert subset_a != subset_c


# ---------------------------------------------------------------------------
# 4. Hydra CLI: skip_model_load is a declared key
# ---------------------------------------------------------------------------


class TestInferenceCLIContract:
    def _compose(self, config_name: str, overrides: list[str]):
        """Compose a Hydra config the way inference.py does, without running it."""
        from hydra import compose, initialize_config_dir

        repo_root = Path(__file__).resolve().parents[1]
        config_dir = repo_root / "src" / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

    def test_inference_gqa_skip_model_load_bare_override(self):
        cfg = self._compose("inference_gqa", ["inferencer.skip_model_load=True"])
        assert cfg.inferencer.skip_model_load is True

    def test_inference_gqa_default_skip_is_false(self):
        cfg = self._compose("inference_gqa", [])
        assert cfg.inferencer.skip_model_load is False


# ---------------------------------------------------------------------------
# 5. Backward-compat import surface
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_model_exports_preserved(self):
        """Supported GQA and backward-compat model classes remain importable."""
        from src.model import (
            DenseGATLayer,
            GQAVQAGNNModel,
            HFQuestionEncoder,
            VQAGNNModel,
        )

        assert GQAVQAGNNModel is not None
        assert VQAGNNModel is not None  # kept for backward compatibility
        assert DenseGATLayer is not None
        assert HFQuestionEncoder is not None

    def test_dataset_exports_preserved(self):
        from src.datasets import GQADataset, GQADemoDataset, VQADataset, VQADemoDataset

        assert GQADataset is not None
        assert GQADemoDataset is not None
        assert VQADataset is not None
        assert VQADemoDataset is not None

    def test_vqa_loss_and_metric_exports_preserved(self):
        from src.loss import VQALoss
        from src.metrics import VQAAccuracy

        assert VQALoss is not None
        assert VQAAccuracy is not None

    def test_dense_gat_layer_reexport_consistent(self):
        """`DenseGATLayer` must be the same class whether imported from the shared
        core or the monolithic `vqa_gnn` module — tests still import it from
        `src.model.vqa_gnn`."""
        from src.model.gnn_core import DenseGATLayer as core
        from src.model.vqa_gnn import DenseGATLayer as reexported

        assert core is reexported, "DenseGATLayer must be a single class, re-exported"


# ---------------------------------------------------------------------------
# 8. GQA runtime config contract: relation vocab and model wiring
# ---------------------------------------------------------------------------


class TestGQAModelConfigContract:
    """Lock the runtime GQA model config so it stays paper-aligned.

    The strict GQA package validated against this repo (uploaded to Kaggle)
    stores `graph_edge_type_count = 624`. `DenseGATLayer` raises ValueError
    if any edge id in the batch is >= num_relations, so the model config
    MUST cover the relation vocab. A regression here silently breaks the
    first training/inference batch — a unit-test guard is cheap.
    """

    EXPECTED_RELATION_COUNT = 624

    def _compose(self, config_name: str, overrides: list[str]):
        from hydra import compose, initialize_config_dir

        repo_root = Path(__file__).resolve().parents[1]
        config_dir = repo_root / "src" / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

    def test_baseline_gqa_targets_gqa_model(self):
        """`baseline_gqa` must instantiate `GQAVQAGNNModel`, not the monolithic class."""
        cfg = self._compose("baseline_gqa", [])
        assert cfg.model._target_ == "src.model.GQAVQAGNNModel"

    def test_baseline_gqa_num_relations_covers_vocab(self):
        """`baseline_gqa` must set `model.num_relations >= 624` (validated vocab)."""
        cfg = self._compose("baseline_gqa", [])
        assert cfg.model.num_relations >= self.EXPECTED_RELATION_COUNT, (
            f"baseline_gqa.model.num_relations={cfg.model.num_relations} "
            f"is below the validated relation vocab "
            f"({self.EXPECTED_RELATION_COUNT}); the first batch with edge ids "
            f">= num_relations would crash DenseGATLayer."
        )

    def test_inference_gqa_num_relations_covers_vocab(self):
        cfg = self._compose("inference_gqa", [])
        assert cfg.model.num_relations >= self.EXPECTED_RELATION_COUNT
