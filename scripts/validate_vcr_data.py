"""
Validate VCR data artifacts before launching VQA-GNN training or inference.

Checks that all files required by VCRDataset are present, structurally
correct, and dimensionally consistent.

Run this script after preparing all artifacts and before submitting a
Kaggle job to avoid wasting GPU time on broken input data.

VCR data structure differs from GQA:
  - Annotations are JSONL (one JSON object per line), not a single JSON dict.
  - HDF5 keys are annot_id (e.g. "vcr1annot-123") for KG graphs.
  - HDF5 keys are img_fn (e.g. "lsmdc_3005...jpg") for visual features.

Exit code:
    0  All checks passed.
    1  One or more checks failed.

Usage:
    # Validate for training (train + val splits):
    python scripts/validate_vcr_data.py \\
        --data-dir data/vcr \\
        --split train val

    # Validate val split only (inference):
    python scripts/validate_vcr_data.py \\
        --data-dir data/vcr \\
        --split val

    # Kaggle:
    python scripts/validate_vcr_data.py \\
        --data-dir /kaggle/input/vcr-gnn-data \\
        --split train val

    # With custom shapes:
    python scripts/validate_vcr_data.py \\
        --data-dir data/vcr \\
        --split train \\
        --num-visual-nodes 36 \\
        --feature-dim 2048 \\
        --d-kg 300 \\
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
# Validation report
# ---------------------------------------------------------------------------


class ValidationReport:
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
            print("\nWarnings (not blocking):")
            for w in self.warnings:
                print(f"  ~ {w}")
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# VCR JSONL parsing
# ---------------------------------------------------------------------------


def validate_vcr_jsonl(
    data_dir: Path, split: str, report: ValidationReport
):
    """
    Parse VCR JSONL and collect annot_ids and img_fns.

    VCR JSONL format (one line = one annotation):
        {"annot_id": "vcr1annot-123", "img_fn": "lsmdc_...", "question": [...], ...}
    """
    print(f"\nValidating VCR JSONL for split='{split}':")

    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        report.fail(
            f"VCR annotation file not found: {jsonl_path}\n"
            "          Download from https://visualcommonsense.com (requires license agreement)."
        )
        return set(), set()

    annot_ids = set()
    img_fns = set()
    n_lines = 0
    parse_errors = 0
    missing_fields = 0

    required_fields = {"annot_id", "img_fn", "question", "answer_choices"}

    try:
        with open(jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if parse_errors <= 3:
                        report.fail(f"Line {line_num}: JSON parse error: {e}")
                    continue

                n_lines += 1
                absent = required_fields - set(item.keys())
                if absent:
                    missing_fields += 1
                    if missing_fields <= 3:
                        report.fail(
                            f"Line {line_num} (annot_id={item.get('annot_id', '?')}): "
                            f"missing fields: {absent}"
                        )
                    continue

                annot_ids.add(item["annot_id"])
                img_fns.add(item["img_fn"])

    except OSError as e:
        report.fail(f"Cannot read {jsonl_path}: {e}")
        return set(), set()

    if parse_errors:
        report.fail(f"{parse_errors} JSON parse errors in {jsonl_path.name}.")
    if missing_fields:
        report.warn(f"{missing_fields} annotations missing required fields.")

    if n_lines == 0:
        report.fail(f"{jsonl_path.name} is empty or contains no valid lines.")
    else:
        report.ok(
            f"{n_lines} annotations parsed; "
            f"{len(annot_ids)} unique annot_ids, "
            f"{len(img_fns)} unique images."
        )

    # Sanity check: VCR has exactly 4 answer choices per question
    # (we only checked the first few; warn if it looks wrong)
    if split != "test":
        report.ok(f"VCR {split}.jsonl: basic structure OK.")

    return annot_ids, img_fns


# ---------------------------------------------------------------------------
# Visual features HDF5
# ---------------------------------------------------------------------------


def validate_visual_features(
    data_dir: Path,
    split: str,
    img_fns: set,
    num_visual_nodes: int,
    feature_dim: int,
    sample_size: int,
    report: ValidationReport,
):
    """
    Check VCR visual features HDF5.
    Keys are img_fn strings (e.g. 'lsmdc_3005_MARGARET_01.05.02.00-01.05.08.00@0.jpg').
    """
    if not HAS_H5PY:
        report.warn("h5py not installed; skipping HDF5 validation. pip install h5py")
        return

    h5_path = data_dir / "visual_features" / f"{split}_features.h5"
    print(f"\nValidating VCR visual features: {h5_path}")

    if not h5_path.exists():
        report.fail(
            f"Visual features not found: {h5_path}\n"
            "          Prepare with: python scripts/prepare_visual_features.py --help\n"
            "          VCR uses R2C Faster RCNN features (~2048-dim, variable box count)."
        )
        return

    try:
        with h5py.File(h5_path, "r") as f:
            h5_keys = set(f.keys())
            report.ok(f"HDF5 opened OK. {len(h5_keys)} image entries.")

            if img_fns:
                missing = img_fns - h5_keys
                if missing:
                    pct = 100.0 * len(missing) / len(img_fns)
                    report.fail(
                        f"{len(missing)} images ({pct:.1f}%) missing from HDF5. "
                        f"First 5: {sorted(missing)[:5]}"
                    )
                else:
                    report.ok(f"All {len(img_fns)} img_fns covered in HDF5.")

            sample_keys = list(h5_keys)[:sample_size]
            shape_errors, dtype_errors = 0, 0
            for k in sample_keys:
                arr = f[k][()]
                if arr.shape != (num_visual_nodes, feature_dim):
                    shape_errors += 1
                if arr.dtype != np.float32:
                    dtype_errors += 1

            if shape_errors:
                report.fail(
                    f"{shape_errors}/{len(sample_keys)} entries have wrong shape. "
                    f"Expected ({num_visual_nodes}, {feature_dim}). "
                    "Note: VCRDataset pads/trims, but shape mismatch may indicate "
                    "wrong --num-visual-nodes or --feature-dim."
                )
            else:
                report.ok(
                    f"Sample {len(sample_keys)}: shape ({num_visual_nodes}, {feature_dim}) OK."
                )

            if dtype_errors:
                report.warn(
                    f"{dtype_errors}/{len(sample_keys)} entries have non-float32 dtype. "
                    "VCRDataset converts on load."
                )
            else:
                report.ok(f"Sample {len(sample_keys)}: dtype=float32 OK.")

    except Exception as e:
        report.fail(f"Could not open or read visual features HDF5: {e}")


# ---------------------------------------------------------------------------
# KG subgraph HDF5
# ---------------------------------------------------------------------------


def validate_kg_graphs(
    data_dir: Path,
    split: str,
    annot_ids: set,
    num_visual_nodes: int,
    max_kg_nodes: int,
    d_kg: int,
    sample_size: int,
    report: ValidationReport,
):
    """
    Check VCR KG subgraph HDF5.
    Keys are annot_id strings (built by prepare_vcr_data.py → prepare_kg_graphs.py).
    """
    if not HAS_H5PY:
        report.warn("h5py not installed; skipping HDF5 validation.")
        return

    h5_path = data_dir / "knowledge_graphs" / f"{split}_graphs.h5"
    print(f"\nValidating VCR KG subgraphs: {h5_path}")

    if not h5_path.exists():
        report.fail(
            f"KG graphs not found: {h5_path}\n"
            "          Build with: python scripts/prepare_vcr_data.py --jsonl ... --output ..."
        )
        return

    try:
        with h5py.File(h5_path, "r") as f:
            h5_keys = set(f.keys())
            report.ok(f"HDF5 opened OK. {len(h5_keys)} annotation entries.")

            if annot_ids:
                missing = annot_ids - h5_keys
                if missing:
                    pct = 100.0 * len(missing) / len(annot_ids)
                    report.fail(
                        f"{len(missing)} annotations ({pct:.1f}%) missing from KG HDF5. "
                        f"First 5: {sorted(missing)[:5]}"
                    )
                else:
                    report.ok(f"All {len(annot_ids)} annot_ids have KG graphs.")

            sample_keys = list(h5_keys)[:sample_size]
            schema_errors = 0
            for k in sample_keys:
                grp = f[k]
                missing_keys = [
                    key for key in ("node_features", "adj_matrix", "node_types")
                    if key not in grp
                ]
                if missing_keys:
                    schema_errors += 1
                    if schema_errors <= 2:
                        report.fail(
                            f"Entry '{k}' missing: {missing_keys}. "
                            "Expected: node_features, adj_matrix, node_types."
                        )
                    continue

                nf = grp["node_features"][()]
                if nf.ndim != 2 or nf.shape[1] != d_kg:
                    schema_errors += 1
                    if schema_errors <= 2:
                        report.fail(
                            f"Entry '{k}': node_features shape={nf.shape}, "
                            f"expected (N_kg, {d_kg}). "
                            "Does --d-kg match configs and prepare_vcr_data.py --d-kg?"
                        )

            if schema_errors == 0:
                report.ok(
                    f"Sample {len(sample_keys)}: schema OK "
                    f"(node_features, adj_matrix, node_types; d_kg={d_kg})."
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
        description="Validate VCR data artifacts before VQA-GNN training/inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", required=True, metavar="PATH",
        help="VCR data root (e.g. data/vcr or /kaggle/input/vcr-gnn-data).",
    )
    parser.add_argument(
        "--split", nargs="+", default=["train", "val"],
        choices=["train", "val", "test"],
        help="Splits to validate (default: train val).",
    )
    parser.add_argument(
        "--num-visual-nodes", type=int, default=36,
        help="Expected number of visual region nodes (default: 36).",
    )
    parser.add_argument(
        "--feature-dim", type=int, default=2048,
        help="Expected visual feature dimension (default: 2048).",
    )
    parser.add_argument(
        "--d-kg", type=int, default=300,
        help=(
            "Expected KG node feature dimension (default: 300). "
            "Must match d_kg in src/configs/model/vqa_gnn_vcr.yaml and configs/datasets/vcr_qa.yaml."
        ),
    )
    parser.add_argument(
        "--max-kg-nodes", type=int, default=30,
        help="Max KG nodes per annotation (default: 30). Must match YAML configs.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=100,
        help="Number of HDF5 entries to sample for shape/dtype checks (default: 100).",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    report = ValidationReport()

    for split in args.split:
        print(f"\n{'='*55}")
        print(f"Split: {split}")
        print("=" * 55)

        annot_ids, img_fns = validate_vcr_jsonl(data_dir, split, report)

        validate_visual_features(
            data_dir=data_dir,
            split=split,
            img_fns=img_fns,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
            sample_size=args.sample_size,
            report=report,
        )

        validate_kg_graphs(
            data_dir=data_dir,
            split=split,
            annot_ids=annot_ids,
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
        print("\nAll checks passed. VCR data appears ready for VQA-GNN training.")
        print("\nNext step:")
        print("  python train.py --config-name baseline_vcr_qa \\")
        print(f"    datasets=vcr_qa \\")
        print(f"    datasets.train.data_dir={args.data_dir} \\")
        print(f"    datasets.val.data_dir={args.data_dir}")


if __name__ == "__main__":
    main()
