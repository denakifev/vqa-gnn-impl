"""
Validate VQA data artifacts before launching training or inference.

Checks that all required data files are present, structurally correct,
and compatible with the VQA-GNN dataset interface.

Run this script after preparing all artifacts and before submitting a
Kaggle job to avoid wasting GPU time on broken input data.

Exit code:
    0  All checks passed.
    1  One or more checks failed (errors printed to stderr).

Usage:
    # Validate for training (train + val splits):
    python scripts/validate_data.py \\
        --data-dir data/vqa \\
        --split train val \\
        --answer-vocab data/vqa/answer_vocab.json

    # Validate val split only (inference):
    python scripts/validate_data.py \\
        --data-dir data/vqa \\
        --split val \\
        --answer-vocab data/vqa/answer_vocab.json

    # Kaggle:
    python scripts/validate_data.py \\
        --data-dir /kaggle/input/vqa-gnn-data \\
        --split train val \\
        --answer-vocab /kaggle/input/vqa-gnn-data/answer_vocab.json

    # With custom expected shapes:
    python scripts/validate_data.py \\
        --data-dir data/vqa \\
        --split train \\
        --answer-vocab data/vqa/answer_vocab.json \\
        --num-visual-nodes 36 \\
        --feature-dim 2048 \\
        --d-kg 300 \\
        --num-answers 3129 \\
        --max-kg-nodes 30 \\
        --sample-size 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ValidationReport:
    """Accumulates pass/fail results and prints a summary."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passes = []

    def ok(self, msg):
        self.passes.append(msg)
        print(f"  [OK]      {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN]    {msg}", file=sys.stderr)

    def fail(self, msg):
        self.errors.append(msg)
        print(f"  [FAIL]    {msg}", file=sys.stderr)

    def summary(self):
        print("\n--- Validation Summary ---")
        print(f"  Passed:   {len(self.passes)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors:   {len(self.errors)}")
        if self.errors:
            print("\nErrors to fix:")
            for e in self.errors:
                print(f"  * {e}")
        if self.warnings:
            print("\nWarnings (not blocking, but should be reviewed):")
            for w in self.warnings:
                print(f"  ~ {w}")
        return len(self.errors) == 0


def _load_json_safe(path, report: ValidationReport, description: str):
    """Load JSON with clear error messages."""
    path = Path(path)
    if not path.exists():
        report.fail(f"{description} not found: {path}")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        report.ok(f"{description} loaded: {path}")
        return data
    except json.JSONDecodeError as e:
        report.fail(f"{description} JSON parse error in {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Answer vocabulary checks
# ---------------------------------------------------------------------------


def validate_answer_vocab(vocab_path: str, expected_n: int, report: ValidationReport):
    """
    Check answer_vocab.json structure.

    Expects: {"answer_to_idx": {"yes": 0, "no": 1, ...}}
    """
    print(f"\nValidating answer vocabulary: {vocab_path}")
    data = _load_json_safe(vocab_path, report, "answer_vocab.json")
    if data is None:
        return None

    if "answer_to_idx" not in data:
        report.fail(
            f"answer_vocab.json missing key 'answer_to_idx'. "
            f"Found keys: {list(data.keys())}"
        )
        return None

    vocab = data["answer_to_idx"]
    n = len(vocab)
    report.ok(f"answer_to_idx has {n} entries.")

    if n != expected_n:
        report.warn(
            f"Vocabulary size {n} differs from expected {expected_n}. "
            "Update num_answers in src/configs/model/vqa_gnn.yaml if you changed top-N."
        )

    # Check index range
    idx_values = list(vocab.values())
    if not all(isinstance(v, int) for v in idx_values):
        report.fail("Some values in answer_to_idx are not integers.")
    else:
        min_idx = min(idx_values)
        max_idx = max(idx_values)
        if min_idx != 0 or max_idx != n - 1:
            report.fail(
                f"Index range [{min_idx}, {max_idx}] is not contiguous [0, {n-1}]. "
                "Vocabulary is corrupted."
            )
        else:
            report.ok(f"Index range [0, {n-1}] is contiguous.")

    return vocab


# ---------------------------------------------------------------------------
# Question/annotation JSON checks
# ---------------------------------------------------------------------------


def validate_question_files(
    data_dir: Path, split: str, report: ValidationReport
):
    """Check question and annotation JSON files."""
    print(f"\nValidating question files for split='{split}':")

    q_path = data_dir / "questions" / f"{split}_questions.json"
    data = _load_json_safe(q_path, report, f"{split} questions")
    question_ids = set()
    image_ids = set()
    if data is not None:
        questions = data.get("questions", [])
        if not questions:
            report.fail(f"No 'questions' key or empty list in {q_path}")
        else:
            report.ok(f"{len(questions)} questions found.")
            for q in questions:
                question_ids.add(str(q["question_id"]))
                image_ids.add(str(q["image_id"]))

    if split == "test":
        return question_ids, image_ids  # test split has no annotations

    ann_path = data_dir / "annotations" / f"{split}_annotations.json"
    ann_data = _load_json_safe(ann_path, report, f"{split} annotations")
    if ann_data is not None:
        annotations = ann_data.get("annotations", [])
        if not annotations:
            report.fail(f"No 'annotations' key or empty list in {ann_path}")
        else:
            report.ok(f"{len(annotations)} annotations found.")
            ann_qids = {str(a["question_id"]) for a in annotations}
            missing_anns = question_ids - ann_qids
            if missing_anns:
                pct = 100.0 * len(missing_anns) / max(len(question_ids), 1)
                report.warn(
                    f"{len(missing_anns)} questions ({pct:.1f}%) have no annotation. "
                    "These will be skipped at training time."
                )
            else:
                report.ok("All questions have annotations.")

    return question_ids, image_ids


# ---------------------------------------------------------------------------
# Visual features HDF5 checks
# ---------------------------------------------------------------------------


def validate_visual_features(
    data_dir: Path,
    split: str,
    image_ids: set,
    num_visual_nodes: int,
    feature_dim: int,
    sample_size: int,
    report: ValidationReport,
):
    """Check visual features HDF5 file."""
    if not HAS_H5PY:
        report.warn("h5py not installed; skipping HDF5 validation. pip install h5py")
        return

    h5_path = data_dir / "visual_features" / f"{split}_features.h5"
    print(f"\nValidating visual features: {h5_path}")

    if not h5_path.exists():
        report.fail(
            f"Visual features not found: {h5_path}\n"
            "          Run: python scripts/prepare_visual_features.py --help"
        )
        return

    try:
        with h5py.File(h5_path, "r") as f:
            h5_keys = set(f.keys())
            report.ok(f"HDF5 opened OK. {len(h5_keys)} image entries.")

            # Coverage check
            if image_ids:
                missing = image_ids - h5_keys
                if missing:
                    pct = 100.0 * len(missing) / len(image_ids)
                    report.fail(
                        f"{len(missing)} images ({pct:.1f}%) are missing from HDF5. "
                        f"Example missing: {sorted(missing)[:5]}"
                    )
                else:
                    report.ok(f"All {len(image_ids)} image_ids have features.")

            # Shape and dtype sample check
            sample_keys = list(h5_keys)[:sample_size]
            shape_errors = 0
            dtype_errors = 0
            for k in sample_keys:
                arr = f[k][()]
                expected_shape = (num_visual_nodes, feature_dim)
                if arr.shape != expected_shape:
                    shape_errors += 1
                if arr.dtype != np.float32:
                    dtype_errors += 1

            if shape_errors:
                report.fail(
                    f"{shape_errors}/{len(sample_keys)} sampled entries have wrong shape. "
                    f"Expected ({num_visual_nodes}, {feature_dim}). "
                    "Update --num-boxes and/or --feature-dim flags."
                )
            else:
                report.ok(
                    f"Sample of {len(sample_keys)}: shape=({num_visual_nodes}, {feature_dim}) OK."
                )

            if dtype_errors:
                report.warn(
                    f"{dtype_errors}/{len(sample_keys)} entries have non-float32 dtype. "
                    "VQADataset converts on load, but this may slow things down."
                )
            else:
                report.ok(f"Sample of {len(sample_keys)}: dtype=float32 OK.")

    except Exception as e:
        report.fail(f"Could not open or read visual features HDF5: {e}")


# ---------------------------------------------------------------------------
# Knowledge graph HDF5 checks
# ---------------------------------------------------------------------------


def validate_kg_graphs(
    data_dir: Path,
    split: str,
    question_ids: set,
    num_visual_nodes: int,
    max_kg_nodes: int,
    d_kg: int,
    sample_size: int,
    report: ValidationReport,
):
    """Check knowledge graph HDF5 file."""
    if not HAS_H5PY:
        report.warn("h5py not installed; skipping HDF5 validation.")
        return

    h5_path = data_dir / "knowledge_graphs" / f"{split}_graphs.h5"
    print(f"\nValidating KG subgraphs: {h5_path}")

    if not h5_path.exists():
        report.fail(
            f"KG graphs not found: {h5_path}\n"
            "          Run: python scripts/prepare_kg_graphs.py --help"
        )
        return

    n_total_expected = num_visual_nodes + 1 + max_kg_nodes

    try:
        with h5py.File(h5_path, "r") as f:
            h5_keys = set(f.keys())
            report.ok(f"HDF5 opened OK. {len(h5_keys)} question entries.")

            # Coverage check
            if question_ids:
                missing = question_ids - h5_keys
                if missing:
                    pct = 100.0 * len(missing) / len(question_ids)
                    report.fail(
                        f"{len(missing)} questions ({pct:.1f}%) are missing from KG HDF5. "
                        f"Example missing: {sorted(missing)[:5]}"
                    )
                else:
                    report.ok(f"All {len(question_ids)} question_ids have KG graphs.")

            # Schema check on sample
            sample_keys = list(h5_keys)[:sample_size]
            schema_errors = 0

            for k in sample_keys:
                grp = f[k]
                missing_keys = []
                for expected_key in ("node_features", "adj_matrix", "node_types"):
                    if expected_key not in grp:
                        missing_keys.append(expected_key)

                if missing_keys:
                    schema_errors += 1
                    if schema_errors == 1:
                        report.fail(
                            f"Question '{k}' missing dataset keys: {missing_keys}. "
                            "Expected: node_features, adj_matrix, node_types."
                        )
                    continue

                # Shape checks (lenient: dataset code handles padding/trimming)
                nf = grp["node_features"][()]
                adj = grp["adj_matrix"][()]
                nt = grp["node_types"][()]

                if nf.ndim != 2 or nf.shape[1] != d_kg:
                    schema_errors += 1
                    if schema_errors <= 3:
                        report.fail(
                            f"Question '{k}': node_features shape={nf.shape}, "
                            f"expected (N_kg, {d_kg}). "
                            "Update --d-kg or configs to match."
                        )

                n_total_stored = nt.shape[0]
                n_kg_stored = nf.shape[0]
                n_total_inferred = num_visual_nodes + 1 + n_kg_stored
                if adj.shape != (n_total_inferred, n_total_inferred):
                    schema_errors += 1
                    if schema_errors <= 3:
                        report.warn(
                            f"Question '{k}': adj shape={adj.shape}, "
                            f"inferred size={n_total_inferred}. "
                            "Dataset code pads/trims, but this should match."
                        )

            if schema_errors == 0:
                report.ok(
                    f"Sample of {len(sample_keys)}: schema (node_features, adj_matrix, "
                    f"node_types) with d_kg={d_kg} OK."
                )
            else:
                report.fail(f"{schema_errors} schema errors in sample of {len(sample_keys)}.")

    except Exception as e:
        report.fail(f"Could not open or read KG HDF5: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Validate VQA data artifacts before training/inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="PATH",
        help="Root data directory (e.g. data/vqa or /kaggle/input/vqa-gnn-data).",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Splits to validate (default: train val).",
    )
    parser.add_argument(
        "--answer-vocab",
        metavar="PATH",
        help=(
            "Path to answer_vocab.json. "
            "Defaults to <data-dir>/answer_vocab.json."
        ),
    )
    parser.add_argument(
        "--num-visual-nodes",
        type=int,
        default=36,
        help="Expected number of visual region nodes (default: 36).",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=2048,
        help="Expected visual feature dimension (default: 2048).",
    )
    parser.add_argument(
        "--d-kg",
        type=int,
        default=100,
        help=(
            "Expected KG node feature dimension (default: 100, matching YAML config defaults). "
            "Must match d_kg in YAML configs and the value used in prepare_kg_graphs.py. "
            "Use 300 for full ConceptNet Numberbatch (requires updating YAML configs too)."
        ),
    )
    parser.add_argument(
        "--max-kg-nodes",
        type=int,
        default=30,
        help="Max KG nodes per question (default: 30). Must match YAML configs.",
    )
    parser.add_argument(
        "--num-answers",
        type=int,
        default=3129,
        help="Expected answer vocabulary size (default: 3129).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of entries to sample for shape/dtype checks (default: 100).",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    report = ValidationReport()

    # Validate answer vocabulary
    vocab_path = args.answer_vocab or str(data_dir / "answer_vocab.json")
    validate_answer_vocab(vocab_path, args.num_answers, report)

    # Validate per-split artifacts
    for split in args.split:
        print(f"\n{'='*50}")
        print(f"Split: {split}")
        print("=" * 50)

        question_ids, image_ids = validate_question_files(data_dir, split, report)

        validate_visual_features(
            data_dir=data_dir,
            split=split,
            image_ids=image_ids,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
            sample_size=args.sample_size,
            report=report,
        )

        validate_kg_graphs(
            data_dir=data_dir,
            split=split,
            question_ids=question_ids,
            num_visual_nodes=args.num_visual_nodes,
            max_kg_nodes=args.max_kg_nodes,
            d_kg=args.d_kg,
            sample_size=args.sample_size,
            report=report,
        )

    success = report.summary()

    if not success:
        print(
            "\nValidation FAILED. Fix the errors above before launching training.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("\nAll checks passed. Data appears ready for training.")


if __name__ == "__main__":
    main()
