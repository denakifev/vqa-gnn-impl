"""
Validate processed VQA-2 artifacts before training or inference.

Checks:
  - answer vocabulary size and contiguous indices
  - VQA-2 question / annotation JSON structure
  - visual-feature HDF5 coverage, shape, and dtype
  - KG HDF5 coverage and schema
  - optional runtime compatibility:
    VQADataset -> vqa_collate_fn -> VQAGNNModel -> VQALoss -> VQAAccuracy

VQA-2 is a coursework extension in this repository. Passing this validator is
not a paper-result claim; it only means the package matches the active runtime
contract.

Usage:
    python scripts/validate_vqa_data.py \\
        --data-dir data/vqa \\
        --split train val \\
        --answer-vocab data/vqa/answer_vocab.json \\
        --text-encoder data/roberta-large

    # File/schema-only validation:
    python scripts/validate_vqa_data.py \\
        --data-dir data/vqa \\
        --split train val \\
        --skip-runtime-check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

from src.datasets.vqa_dataset import VQADataset
from src.loss.vqa_loss import VQALoss
from src.metrics.vqa_metric import VQAAccuracy
from src.model.vqa_gnn import VQAGNNModel


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passes: list[str] = []

    def ok(self, message: str) -> None:
        self.passes.append(message)
        print(f"  [OK]      {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  [WARN]    {message}", file=sys.stderr)

    def fail(self, message: str) -> None:
        self.errors.append(message)
        print(f"  [FAIL]    {message}", file=sys.stderr)

    def summary(self) -> bool:
        print("\n--- Validation Summary ---")
        print(f"  Passed:   {len(self.passes)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors:   {len(self.errors)}")
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        return not self.errors


def _load_json(path: str | Path, report: ValidationReport, description: str) -> dict | None:
    path = Path(path)
    if not path.exists():
        report.fail(f"{description} not found: {path}")
        return None
    try:
        with open(path) as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        report.fail(f"{description} could not be parsed: {exc}")
        return None
    report.ok(f"{description} loaded: {path}")
    return payload


def validate_answer_vocab(
    vocab_path: str | Path,
    expected_size: int,
    report: ValidationReport,
) -> dict[str, int] | None:
    print(f"\nValidating answer vocab: {vocab_path}")
    payload = _load_json(vocab_path, report, "answer_vocab.json")
    if payload is None:
        return None

    answer_to_idx = payload.get("answer_to_idx")
    if not isinstance(answer_to_idx, dict):
        report.fail("answer_vocab.json must contain dict key 'answer_to_idx'.")
        return None

    idx_values = list(answer_to_idx.values())
    if not idx_values:
        report.fail("answer_to_idx is empty.")
        return None
    if not all(isinstance(value, int) for value in idx_values):
        report.fail("answer_to_idx contains non-integer indices.")
        return None

    size = len(answer_to_idx)
    if size == expected_size:
        report.ok(f"Answer vocab size matches expected VQA-2 default: {size}.")
    else:
        report.warn(
            f"Answer vocab has {size} entries, expected {expected_size}. "
            "If intentional, update model.num_answers and docs."
        )

    if sorted(idx_values) != list(range(size)):
        report.fail("Answer vocab indices are not contiguous [0..N-1].")
    else:
        report.ok("Answer vocab indices are contiguous.")

    idx_to_answer = payload.get("idx_to_answer")
    if idx_to_answer is None:
        report.warn("idx_to_answer is missing; runtime can continue, but metadata is incomplete.")
    elif isinstance(idx_to_answer, list) and len(idx_to_answer) == size:
        report.ok("idx_to_answer is present and aligned with answer_to_idx.")
    else:
        report.warn("idx_to_answer is present but not aligned with answer_to_idx.")

    return answer_to_idx


def validate_questions_and_annotations(
    data_dir: str | Path,
    split: str,
    answer_to_idx: dict[str, int] | None,
    report: ValidationReport,
) -> tuple[set[str], set[str]]:
    print(f"\nValidating VQA-2 JSON files for split='{split}'")
    data_dir = Path(data_dir)
    question_path = data_dir / "questions" / f"{split}_questions.json"
    question_payload = _load_json(question_path, report, f"{split}_questions.json")
    question_ids: set[str] = set()
    image_ids: set[str] = set()

    if question_payload is not None:
        questions = question_payload.get("questions")
        if not isinstance(questions, list) or not questions:
            report.fail(f"{question_path.name} must contain a non-empty 'questions' list.")
        else:
            malformed = 0
            for item in questions:
                if not all(key in item for key in ("question_id", "image_id", "question")):
                    malformed += 1
                    continue
                question_ids.add(str(item["question_id"]))
                image_ids.add(str(item["image_id"]))
            if malformed:
                report.fail(f"{split}: {malformed} questions are missing required keys.")
            report.ok(f"{split}: {len(question_ids)} questions, {len(image_ids)} unique images.")

    if split == "test":
        return question_ids, image_ids

    annotation_path = data_dir / "annotations" / f"{split}_annotations.json"
    annotation_payload = _load_json(annotation_path, report, f"{split}_annotations.json")
    if annotation_payload is None:
        return question_ids, image_ids

    annotations = annotation_payload.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        report.fail(f"{annotation_path.name} must contain a non-empty 'annotations' list.")
        return question_ids, image_ids

    ann_question_ids = set()
    malformed = 0
    no_vocab_hit = 0
    for ann in annotations:
        if "question_id" not in ann or "answers" not in ann:
            malformed += 1
            continue
        ann_question_ids.add(str(ann["question_id"]))
        answers = ann.get("answers", [])
        if not isinstance(answers, list) or not answers:
            malformed += 1
            continue
        if answer_to_idx is not None:
            # Import here to keep this validator locked to the dataset's
            # runtime normalization.
            from src.datasets.vqa_dataset import _normalize_vqa_answer

            if not any(_normalize_vqa_answer(a.get("answer", "")) in answer_to_idx for a in answers):
                no_vocab_hit += 1

    if malformed:
        report.fail(f"{split}: {malformed} annotations are malformed.")
    else:
        report.ok(f"{split}: {len(annotations)} annotations have required keys.")

    missing_annotations = question_ids - ann_question_ids
    if missing_annotations:
        pct = 100.0 * len(missing_annotations) / max(len(question_ids), 1)
        report.warn(f"{split}: {len(missing_annotations)} questions ({pct:.2f}%) have no annotation.")
    else:
        report.ok(f"{split}: all questions have annotations.")

    if no_vocab_hit:
        pct = 100.0 * no_vocab_hit / max(len(annotations), 1)
        report.warn(f"{split}: {no_vocab_hit} annotations ({pct:.2f}%) have no answers in vocab.")
    else:
        report.ok(f"{split}: sampled annotation answers map into the answer vocab.")

    return question_ids, image_ids


def validate_visual_features(
    data_dir: str | Path,
    split: str,
    expected_image_ids: set[str],
    num_visual_nodes: int,
    feature_dim: int,
    sample_size: int,
    report: ValidationReport,
) -> None:
    print(f"\nValidating visual features for split='{split}'")
    if h5py is None:
        report.fail("h5py is required to validate visual feature HDF5 files.")
        return

    path = Path(data_dir) / "visual_features" / f"{split}_features.h5"
    if not path.exists():
        report.fail(f"Visual feature HDF5 not found: {path}")
        return

    try:
        with h5py.File(path, "r") as handle:
            keys = set(handle.keys())
            if not keys:
                report.fail(f"{path} has no image entries.")
                return
            report.ok(f"{split}: visual HDF5 opened with {len(keys)} image entries.")

            missing = expected_image_ids - keys
            if missing:
                report.fail(
                    f"{split}: visual HDF5 is missing {len(missing)} imageIds. "
                    f"Example: {sorted(missing)[:5]}"
                )
            else:
                report.ok(f"{split}: all referenced imageIds have visual features.")

            sample_keys = sorted(keys)[:sample_size]
            shape_errors = 0
            dtype_warnings = 0
            for key in sample_keys:
                array = handle[key][()]
                if array.shape != (num_visual_nodes, feature_dim):
                    shape_errors += 1
                if array.dtype != np.float32:
                    dtype_warnings += 1

            if shape_errors:
                report.fail(
                    f"{split}: {shape_errors}/{len(sample_keys)} sampled visual entries "
                    f"have wrong shape; expected ({num_visual_nodes}, {feature_dim})."
                )
            else:
                report.ok(
                    f"{split}: visual sample shape ({num_visual_nodes}, {feature_dim}) OK."
                )

            if dtype_warnings:
                report.warn(
                    f"{split}: {dtype_warnings}/{len(sample_keys)} visual entries are not float32."
                )
            else:
                report.ok(f"{split}: visual sample dtype float32 OK.")
    except Exception as exc:
        report.fail(f"{split}: could not read visual HDF5: {exc}")


def validate_kg_graphs(
    data_dir: str | Path,
    split: str,
    expected_question_ids: set[str],
    num_visual_nodes: int,
    max_kg_nodes: int,
    d_kg: int,
    sample_size: int,
    report: ValidationReport,
) -> None:
    print(f"\nValidating KG graphs for split='{split}'")
    if h5py is None:
        report.fail("h5py is required to validate KG HDF5 files.")
        return

    path = Path(data_dir) / "knowledge_graphs" / f"{split}_graphs.h5"
    if not path.exists():
        report.fail(f"KG graph HDF5 not found: {path}")
        return

    try:
        with h5py.File(path, "r") as handle:
            keys = set(handle.keys())
            if not keys:
                report.fail(f"{path} has no question entries.")
                return
            report.ok(f"{split}: KG HDF5 opened with {len(keys)} question entries.")

            missing = expected_question_ids - keys
            if missing:
                report.fail(
                    f"{split}: KG HDF5 is missing {len(missing)} questionIds. "
                    f"Example: {sorted(missing)[:5]}"
                )
            else:
                report.ok(f"{split}: all referenced questionIds have KG graphs.")

            sample_keys = sorted(keys)[:sample_size]
            schema_errors = 0
            warnings = 0
            for key in sample_keys:
                group = handle[key]
                missing_members = [
                    member for member in ("node_features", "adj_matrix", "node_types")
                    if member not in group
                ]
                if missing_members:
                    schema_errors += 1
                    if schema_errors <= 3:
                        report.fail(f"{split}/{key}: missing HDF5 datasets {missing_members}.")
                    continue

                node_features = group["node_features"][()]
                adj = group["adj_matrix"][()]
                node_types = group["node_types"][()]

                if node_features.ndim != 2 or node_features.shape[1] != d_kg:
                    schema_errors += 1
                    if schema_errors <= 3:
                        report.fail(
                            f"{split}/{key}: node_features shape={node_features.shape}, "
                            f"expected (N_kg, {d_kg})."
                        )
                if node_features.shape[0] > max_kg_nodes:
                    warnings += 1
                    if warnings <= 3:
                        report.warn(
                            f"{split}/{key}: N_kg={node_features.shape[0]} exceeds "
                            f"max_kg_nodes={max_kg_nodes}; dataset will trim."
                        )

                expected_total = num_visual_nodes + 1 + node_features.shape[0]
                if adj.shape != (expected_total, expected_total):
                    schema_errors += 1
                    if schema_errors <= 3:
                        report.fail(
                            f"{split}/{key}: adj shape={adj.shape}, expected "
                            f"({expected_total}, {expected_total}) from stored N_kg."
                        )
                if node_types.shape != (expected_total,):
                    schema_errors += 1
                    if schema_errors <= 3:
                        report.fail(
                            f"{split}/{key}: node_types shape={node_types.shape}, "
                            f"expected ({expected_total},)."
                        )
                if node_types.size and not set(np.unique(node_types)).issubset({0, 1, 2}):
                    schema_errors += 1
                    if schema_errors <= 3:
                        report.fail(f"{split}/{key}: node_types contains ids outside {{0,1,2}}.")

            if schema_errors == 0:
                report.ok(
                    f"{split}: KG sample schema OK "
                    f"(node_features d_kg={d_kg}, adj, node_types)."
                )
            else:
                report.fail(f"{split}: {schema_errors} KG schema errors in sampled entries.")
    except Exception as exc:
        report.fail(f"{split}: could not read KG HDF5: {exc}")


def validate_runtime_path(
    data_dir: str | Path,
    answer_vocab_path: str | Path,
    split: str,
    num_visual_nodes: int,
    feature_dim: int,
    d_kg: int,
    max_kg_nodes: int,
    num_answers: int,
    text_encoder: str,
    runtime_batch_size: int,
    report: ValidationReport,
) -> None:
    print("\nValidating runtime path (dataset -> collate -> model -> loss -> metric)")
    if torch is None or DataLoader is None:
        report.fail("PyTorch is required for runtime validation.")
        return
    if split == "test":
        report.warn("Runtime validation skipped for test split because labels are unavailable.")
        return

    try:
        dataset = VQADataset(
            partition=split,
            data_dir=str(data_dir),
            answer_vocab_path=str(answer_vocab_path),
            max_question_len=20,
            num_visual_nodes=num_visual_nodes,
            max_kg_nodes=max_kg_nodes,
            d_visual=feature_dim,
            d_kg=d_kg,
            text_encoder_name=text_encoder,
            limit=max(runtime_batch_size, 1),
            shuffle_index=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(runtime_batch_size, len(dataset)),
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        batch = next(iter(loader))
        batch_size = batch["labels"].shape[0]
        expected_total_nodes = num_visual_nodes + 1 + max_kg_nodes

        expected_shapes = {
            "visual_features": (batch_size, num_visual_nodes, feature_dim),
            "graph_node_features": (batch_size, max_kg_nodes, d_kg),
            "graph_adj": (batch_size, expected_total_nodes, expected_total_nodes),
            "graph_node_types": (batch_size, expected_total_nodes),
            "answer_scores": (batch_size, len(dataset.answer_to_idx)),
            "labels": (batch_size,),
        }
        for key, expected_shape in expected_shapes.items():
            if tuple(batch[key].shape) != expected_shape:
                report.fail(
                    f"Runtime batch {key} shape mismatch: "
                    f"{tuple(batch[key].shape)} != {expected_shape}"
                )
                return
        report.ok("Runtime batch shapes match VQA-2 contract.")

        model = VQAGNNModel(
            d_visual=feature_dim,
            d_kg=d_kg,
            d_hidden=512,
            num_gnn_layers=3,
            num_heads=8,
            num_answers=num_answers,
            text_encoder_name=text_encoder,
            freeze_text_encoder=True,
        )
        model.eval()
        criterion = VQALoss(use_soft_labels=True)
        metric = VQAAccuracy(name="VQA_Accuracy")

        with torch.no_grad():
            outputs = model(**batch)
            losses = criterion(**outputs, **batch)
            accuracy = metric(**outputs, **batch)

        logits = outputs["logits"]
        if logits.shape != (batch_size, num_answers):
            report.fail(f"Model logits shape mismatch: {tuple(logits.shape)}")
            return

        loss_value = float(losses["loss"].item())
        if not math.isfinite(loss_value):
            report.fail(f"Loss is not finite: {loss_value}")
            return
        if not math.isfinite(float(accuracy)):
            report.fail(f"Metric is not finite: {accuracy}")
            return

        report.ok(
            f"One-batch forward succeeded: logits={tuple(logits.shape)}, "
            f"loss={loss_value:.6f}, metric={accuracy:.6f}."
        )
    except Exception as exc:
        report.fail(f"Runtime validation failed: {exc}")


def _default_text_encoder() -> str:
    local_path = REPO_ROOT / "data" / "roberta-large"
    if local_path.exists():
        return str(local_path)
    return "roberta-large"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate processed VQA-2 artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--answer-vocab", default=None)
    parser.add_argument("--split", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--num-visual-nodes", type=int, default=36)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--d-kg", type=int, default=300)
    parser.add_argument("--max-kg-nodes", type=int, default=30)
    parser.add_argument("--num-answers", type=int, default=3129)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--text-encoder", default=_default_text_encoder())
    parser.add_argument("--runtime-batch-size", type=int, default=1)
    parser.add_argument("--skip-runtime-check", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    answer_vocab_path = Path(args.answer_vocab) if args.answer_vocab else data_dir / "answer_vocab.json"
    report = ValidationReport()
    answer_to_idx = validate_answer_vocab(answer_vocab_path, args.num_answers, report)

    runtime_split = None
    for split in args.split:
        print(f"\n{'=' * 60}")
        print(f"Split: {split}")
        print("=" * 60)

        question_ids, image_ids = validate_questions_and_annotations(
            data_dir=data_dir,
            split=split,
            answer_to_idx=answer_to_idx,
            report=report,
        )
        validate_visual_features(
            data_dir=data_dir,
            split=split,
            expected_image_ids=image_ids,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
            sample_size=args.sample_size,
            report=report,
        )
        validate_kg_graphs(
            data_dir=data_dir,
            split=split,
            expected_question_ids=question_ids,
            num_visual_nodes=args.num_visual_nodes,
            max_kg_nodes=args.max_kg_nodes,
            d_kg=args.d_kg,
            sample_size=args.sample_size,
            report=report,
        )
        if runtime_split is None and split != "test":
            runtime_split = split

    if not args.skip_runtime_check and answer_to_idx is not None and runtime_split is not None:
        validate_runtime_path(
            data_dir=data_dir,
            answer_vocab_path=answer_vocab_path,
            split=runtime_split,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
            d_kg=args.d_kg,
            max_kg_nodes=args.max_kg_nodes,
            num_answers=args.num_answers,
            text_encoder=args.text_encoder,
            runtime_batch_size=args.runtime_batch_size,
            report=report,
        )
    elif args.skip_runtime_check:
        report.warn("Runtime check skipped by user request.")

    success = report.summary()
    if not success:
        print("\nValidation FAILED.", file=sys.stderr)
        sys.exit(1)

    print("\nValidation passed. VQA-2 data looks ready for runtime use.")


if __name__ == "__main__":
    main()
