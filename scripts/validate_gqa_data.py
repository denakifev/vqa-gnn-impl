"""
Strict validation for processed GQA artifacts.

Checks:
  - answer vocab size / contiguous indices / metadata
  - relation vocab schema / special relations / contiguous indices
  - official balanced question files
  - visual-feature coverage and shape
  - scene-graph HDF5 coverage, schema, attrs, and edge-type matrices
  - optional split-level metadata JSON files
  - runtime compatibility: GQADataset -> collate -> model -> loss -> metric
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

from src.datasets.gqa_dataset import GQADataset
from src.loss.gqa_loss import GQALoss
from src.metrics.gqa_metric import GQAAccuracy
from src.model.gqa_model import GQAVQAGNNModel


EDGE_SPECIALS = ("__no_edge__", "__self__", "__question_visual__", "__question_textual__")


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
        with open(path) as f:
            payload = json.load(f)
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
    payload = _load_json(vocab_path, report, "gqa_answer_vocab.json")
    if payload is None:
        return None

    answer_to_idx = payload.get("answer_to_idx")
    if not isinstance(answer_to_idx, dict):
        report.fail("gqa_answer_vocab.json missing dict key 'answer_to_idx'.")
        return None

    idx_values = list(answer_to_idx.values())
    if not idx_values:
        report.fail("answer_to_idx is empty.")
        return None
    if not all(isinstance(value, int) for value in idx_values):
        report.fail("answer_to_idx contains non-integer indices.")
        return None

    size = len(answer_to_idx)
    if size != expected_size:
        report.warn(
            f"Answer vocab has {size} entries, expected {expected_size}. "
            "If intentional, update model.num_answers and dataset docs accordingly."
        )
    else:
        report.ok(f"Answer vocab size matches expected paper target: {size}.")

    if sorted(idx_values) != list(range(size)):
        report.fail("Answer vocab indices are not contiguous [0..N-1].")
    else:
        report.ok("Answer vocab indices are contiguous.")

    idx_to_answer = payload.get("idx_to_answer")
    if isinstance(idx_to_answer, list) and len(idx_to_answer) == size:
        report.ok("idx_to_answer is present and aligned with answer_to_idx.")
    else:
        report.warn("idx_to_answer missing or misaligned; runtime can continue, but metadata is incomplete.")

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        report.ok(
            f"Answer vocab metadata present: policy={metadata.get('selection_policy', 'unknown')}, "
            f"coverage_pct={metadata.get('coverage_pct', 'unknown')}"
        )
    else:
        report.warn("Answer vocab metadata missing.")

    return answer_to_idx


def validate_relation_vocab(
    relation_vocab_path: str | Path,
    report: ValidationReport,
) -> tuple[dict[str, int], list[str]] | None:
    print(f"\nValidating relation vocab: {relation_vocab_path}")
    payload = _load_json(relation_vocab_path, report, "gqa_relation_vocab.json")
    if payload is None:
        return None

    relation_to_idx = payload.get("relation_to_idx")
    idx_to_relation = payload.get("idx_to_relation")
    if not isinstance(relation_to_idx, dict) or not isinstance(idx_to_relation, list):
        report.fail("gqa_relation_vocab.json must contain relation_to_idx and idx_to_relation.")
        return None

    idx_values = list(relation_to_idx.values())
    size = len(relation_to_idx)
    if sorted(idx_values) != list(range(size)):
        report.fail("Relation vocab indices are not contiguous [0..N-1].")
    else:
        report.ok("Relation vocab indices are contiguous.")

    if len(idx_to_relation) != size:
        report.fail("idx_to_relation length does not match relation_to_idx.")
    else:
        report.ok("idx_to_relation is aligned with relation_to_idx.")

    missing_specials = [special for special in EDGE_SPECIALS if special not in relation_to_idx]
    if missing_specials:
        report.fail(f"Relation vocab is missing required special relations: {missing_specials}")
    else:
        report.ok("Relation vocab contains all required special relations.")

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        report.ok(
            f"Relation vocab metadata present: predicates={metadata.get('predicate_count', 'unknown')}, "
            f"relations={metadata.get('relation_count_total', 'unknown')}"
        )
    else:
        report.warn("Relation vocab metadata missing.")

    return relation_to_idx, idx_to_relation


def validate_questions(
    data_dir: str | Path,
    split: str,
    report: ValidationReport,
) -> tuple[set[str], set[str]]:
    print(f"\nValidating questions for split='{split}'")
    path = Path(data_dir) / "questions" / f"{split}_balanced_questions.json"
    payload = _load_json(path, report, f"{split}_balanced_questions.json")
    if payload is None:
        return set(), set()
    if not isinstance(payload, dict):
        report.fail(f"{path.name} must be an official GQA dict, got {type(payload).__name__}.")
        return set(), set()

    question_ids: set[str] = set()
    image_ids: set[str] = set()
    missing_answer = 0
    for question_id, item in payload.items():
        question_ids.add(str(question_id))
        if "imageId" not in item:
            report.fail(f"Question {question_id} missing imageId.")
            continue
        image_ids.add(str(item["imageId"]))
        if "question" not in item:
            report.fail(f"Question {question_id} missing question text.")
        if split != "test" and not item.get("answer"):
            missing_answer += 1

    report.ok(f"{split}: {len(question_ids)} questions, {len(image_ids)} unique images.")
    if missing_answer:
        report.warn(f"{split}: {missing_answer} questions have no answer field.")
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
            missing = expected_image_ids - keys
            if missing:
                report.fail(
                    f"{split}: visual HDF5 is missing {len(missing)} imageIds. "
                    f"First 5: {sorted(missing)[:5]}"
                )
            else:
                report.ok(f"{split}: all {len(expected_image_ids)} imageIds are covered.")

            attr_num_boxes = handle.attrs.get("num_boxes")
            attr_feature_dim = handle.attrs.get("feature_dim")
            if attr_num_boxes is not None and int(attr_num_boxes) != num_visual_nodes:
                report.fail(
                    f"{split}: visual HDF5 attr num_boxes={attr_num_boxes}, expected {num_visual_nodes}."
                )
            elif attr_num_boxes is not None:
                report.ok(f"{split}: visual HDF5 attr num_boxes={attr_num_boxes}.")
            else:
                report.warn(f"{split}: visual HDF5 has no num_boxes attr.")

            if attr_feature_dim is not None and int(attr_feature_dim) != feature_dim:
                report.fail(
                    f"{split}: visual HDF5 attr feature_dim={attr_feature_dim}, expected {feature_dim}."
                )
            elif attr_feature_dim is not None:
                report.ok(f"{split}: visual HDF5 attr feature_dim={attr_feature_dim}.")
            else:
                report.warn(f"{split}: visual HDF5 has no feature_dim attr.")

            sample_keys = list(keys)[:sample_size]
            wrong_shape = 0
            wrong_dtype = 0
            for key in sample_keys:
                array = handle[key][()]
                if array.shape != (num_visual_nodes, feature_dim):
                    wrong_shape += 1
                if array.dtype != np.float32:
                    wrong_dtype += 1

            if wrong_shape:
                report.fail(
                    f"{split}: {wrong_shape}/{len(sample_keys)} sampled visual entries have wrong shape."
                )
            else:
                report.ok(
                    f"{split}: sampled visual entries have shape ({num_visual_nodes}, {feature_dim})."
                )
            if wrong_dtype:
                report.warn(
                    f"{split}: {wrong_dtype}/{len(sample_keys)} sampled visual entries are not float32."
                )
            else:
                report.ok(f"{split}: sampled visual entries use float32.")
    except Exception as exc:
        report.fail(f"Could not read visual HDF5 {path}: {exc}")


def validate_visual_metadata(
    data_dir: str | Path,
    split: str,
    require_metadata: bool,
    report: ValidationReport,
) -> None:
    metadata_path = Path(data_dir) / "metadata" / f"{split}_visual_features_metadata.json"
    metadata = _load_json(metadata_path, report, f"{split} visual metadata")
    if metadata is None:
        if require_metadata:
            report.fail(f"{split}: required metadata missing at {metadata_path}")
        return

    missing = metadata.get("missing_image_ids", None)
    if isinstance(missing, int) and missing == 0:
        report.ok(f"{split}: visual metadata confirms full image coverage.")
    elif isinstance(missing, int):
        report.fail(f"{split}: visual metadata reports {missing} missing image ids.")
    else:
        report.fail(f"{split}: visual metadata missing 'missing_image_ids'.")


def validate_kg_graphs(
    data_dir: str | Path,
    split: str,
    expected_question_ids: set[str],
    num_visual_nodes: int,
    max_kg_nodes: int,
    d_kg: int,
    relation_vocab_size: int,
    sample_size: int,
    require_metadata: bool,
    report: ValidationReport,
) -> None:
    print(f"\nValidating knowledge graphs for split='{split}'")
    if h5py is None:
        report.fail("h5py is required to validate KG HDF5 files.")
        return

    path = Path(data_dir) / "knowledge_graphs" / f"{split}_graphs.h5"
    if not path.exists():
        report.fail(f"KG HDF5 not found: {path}")
        return

    try:
        with h5py.File(path, "r") as handle:
            keys = set(handle.keys())
            missing = expected_question_ids - keys
            if missing:
                report.fail(
                    f"{split}: KG HDF5 is missing {len(missing)} question_ids. "
                    f"First 5: {sorted(missing)[:5]}"
                )
            else:
                report.ok(f"{split}: all {len(expected_question_ids)} question_ids have graphs.")

            attr_d_kg = handle.attrs.get("d_kg")
            attr_num_visual_nodes = handle.attrs.get("num_visual_nodes")
            attr_max_kg_nodes = handle.attrs.get("max_kg_nodes")
            attr_fc_fallback = handle.attrs.get("fully_connected_fallback_used")
            attr_graph_mode = handle.attrs.get("graph_mode")
            attr_conceptnet_used = handle.attrs.get("conceptnet_used")
            attr_edge_count = handle.attrs.get("graph_edge_type_count")

            if attr_d_kg is not None and int(attr_d_kg) != d_kg:
                report.fail(f"{split}: KG attr d_kg={attr_d_kg}, expected {d_kg}.")
            elif attr_d_kg is not None:
                report.ok(f"{split}: KG attr d_kg={attr_d_kg}.")
            else:
                report.warn(f"{split}: KG HDF5 has no d_kg attr.")

            if attr_num_visual_nodes is not None and int(attr_num_visual_nodes) != num_visual_nodes:
                report.fail(
                    f"{split}: KG attr num_visual_nodes={attr_num_visual_nodes}, expected {num_visual_nodes}."
                )
            elif attr_num_visual_nodes is not None:
                report.ok(f"{split}: KG attr num_visual_nodes={attr_num_visual_nodes}.")
            else:
                report.warn(f"{split}: KG HDF5 has no num_visual_nodes attr.")

            if attr_max_kg_nodes is not None and int(attr_max_kg_nodes) != max_kg_nodes:
                report.fail(
                    f"{split}: KG attr max_kg_nodes={attr_max_kg_nodes}, expected {max_kg_nodes}."
                )
            elif attr_max_kg_nodes is not None:
                report.ok(f"{split}: KG attr max_kg_nodes={attr_max_kg_nodes}.")
            else:
                report.warn(f"{split}: KG HDF5 has no max_kg_nodes attr.")

            if attr_fc_fallback is not None:
                attr_fc_fallback = bool(attr_fc_fallback)
            if attr_fc_fallback is True:
                report.fail(f"{split}: KG HDF5 reports fully_connected_fallback_used=True.")
            elif attr_fc_fallback is False:
                report.ok(f"{split}: KG HDF5 explicitly disables fully-connected fallback.")
            else:
                report.warn(f"{split}: KG HDF5 has no fully_connected_fallback_used attr.")

            if attr_graph_mode == "official_scene_graph":
                report.ok(f"{split}: KG HDF5 graph_mode=official_scene_graph.")
            else:
                report.fail(f"{split}: KG HDF5 graph_mode must be official_scene_graph, got {attr_graph_mode}.")

            if attr_conceptnet_used is False or attr_conceptnet_used == 0:
                report.ok(f"{split}: KG HDF5 confirms ConceptNet is not used.")
            else:
                report.fail(f"{split}: KG HDF5 reports conceptnet_used={attr_conceptnet_used}.")

            if attr_edge_count is not None and int(attr_edge_count) != relation_vocab_size:
                report.fail(
                    f"{split}: KG HDF5 graph_edge_type_count={attr_edge_count}, "
                    f"expected {relation_vocab_size}."
                )
            elif attr_edge_count is not None:
                report.ok(f"{split}: KG HDF5 graph_edge_type_count={attr_edge_count}.")
            else:
                report.warn(f"{split}: KG HDF5 has no graph_edge_type_count attr.")

            sample_keys = list(keys)[:sample_size]
            schema_errors = 0
            fully_connected_visual_blocks = 0
            for key in sample_keys:
                group = handle[key]
                for dataset_name in ("node_features", "adj_matrix", "graph_edge_types", "node_types"):
                    if dataset_name not in group:
                        schema_errors += 1
                        break
                else:
                    node_features = group["node_features"][()]
                    adj = group["adj_matrix"][()]
                    edge_types = group["graph_edge_types"][()]
                    node_types = group["node_types"][()]
                    n_kg = node_features.shape[0]
                    expected_total_nodes = num_visual_nodes + 1 + n_kg

                    if node_features.ndim != 2 or node_features.shape[1] != d_kg:
                        schema_errors += 1
                        continue
                    if n_kg > max_kg_nodes:
                        schema_errors += 1
                        continue
                    if adj.shape != (expected_total_nodes, expected_total_nodes):
                        schema_errors += 1
                        continue
                    if edge_types.shape != (expected_total_nodes, expected_total_nodes):
                        schema_errors += 1
                        continue
                    if node_types.shape != (expected_total_nodes,):
                        schema_errors += 1
                        continue
                    if not np.array_equal(adj > 0, adj.T > 0):
                        schema_errors += 1
                        continue
                    max_edge_type = int(edge_types.max()) if edge_types.size else 0
                    if max_edge_type >= relation_vocab_size:
                        schema_errors += 1
                        continue
                    if np.all(adj[:num_visual_nodes, :num_visual_nodes] == 1.0):
                        fully_connected_visual_blocks += 1

            if schema_errors:
                report.fail(f"{split}: found {schema_errors} schema errors in sampled KG graphs.")
            else:
                report.ok(
                    f"{split}: sampled KG graphs have consistent node_features / adj_matrix / "
                    f"graph_edge_types / node_types."
                )

            if fully_connected_visual_blocks:
                report.fail(
                    f"{split}: {fully_connected_visual_blocks}/{len(sample_keys)} sampled graphs "
                    "have a fully-connected visual block, which indicates a fallback path."
                )
            else:
                report.ok(f"{split}: sampled graphs do not use a fully-connected visual fallback.")
    except Exception as exc:
        report.fail(f"Could not read KG HDF5 {path}: {exc}")
        return

    metadata_path = Path(data_dir) / "metadata" / f"{split}_knowledge_graphs_metadata.json"
    metadata = _load_json(metadata_path, report, f"{split} KG metadata")
    if metadata is None:
        if require_metadata:
            report.fail(f"{split}: required metadata missing at {metadata_path}")
        return

    for key in (
        "questions_without_linked_entities_pct",
        "avg_textual_nodes",
        "avg_visual_nodes",
        "avg_visual_relation_edges",
        "avg_textual_relation_edges",
        "avg_graph_size_total_nodes",
        "fully_connected_fallback_used",
        "graph_mode",
    ):
        if key not in metadata:
            report.fail(f"{split}: KG metadata missing key '{key}'.")

    if metadata.get("graph_mode") == "question_context_plus_visual_textual_scene_graphs":
        report.ok(f"{split}: KG metadata confirms scene-graph graph mode.")
    else:
        report.fail(f"{split}: KG metadata graph_mode is not scene-graph-based.")

    if metadata.get("fully_connected_fallback_used") is True:
        report.fail(f"{split}: KG metadata reports fully_connected_fallback_used=True.")
    elif metadata.get("fully_connected_fallback_used") is False:
        report.ok(f"{split}: KG metadata confirms fully-connected fallback is disabled.")

    if metadata.get("conceptnet_used") is False:
        report.ok(f"{split}: KG metadata confirms ConceptNet is disabled.")
    else:
        report.fail(f"{split}: KG metadata conceptnet_used must be false.")

    report.ok(
        f"{split}: KG metadata no-match rate={metadata.get('questions_without_linked_entities_pct')}%, "
        f"avg_textual_nodes={metadata.get('avg_textual_nodes')}, "
        f"avg_visual_nodes={metadata.get('avg_visual_nodes')}, "
        f"avg_total_nodes={metadata.get('avg_graph_size_total_nodes')}."
    )


def validate_runtime_path(
    data_dir: str | Path,
    answer_vocab_path: str | Path,
    num_visual_nodes: int,
    feature_dim: int,
    d_kg: int,
    max_kg_nodes: int,
    text_encoder: str,
    runtime_batch_size: int,
    relation_vocab_size: int,
    report: ValidationReport,
) -> None:
    print("\nValidating runtime path (dataset -> collate -> model -> loss -> metric)")
    if torch is None or DataLoader is None:
        report.fail("PyTorch is required for runtime validation.")
        return

    try:
        dataset = GQADataset(
            partition="train",
            data_dir=str(data_dir),
            answer_vocab_path=str(answer_vocab_path),
            max_question_len=30,
            num_visual_nodes=num_visual_nodes,
            max_kg_nodes=max_kg_nodes,
            d_visual=feature_dim,
            d_kg=d_kg,
            text_encoder_name=text_encoder,
            limit=max(runtime_batch_size, 1),
            shuffle_index=False,
            strict_answer_vocab=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(runtime_batch_size, len(dataset)),
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        batch = next(iter(loader))
        expected_total_nodes = num_visual_nodes + 1 + max_kg_nodes
        batch_size = batch["labels"].shape[0]

        if any(key in batch for key in ("soft_labels", "answer_label", "n_candidates")):
            report.fail("Runtime batch contains keys from an unsupported task.")
            return

        if batch["visual_features"].shape != (batch_size, num_visual_nodes, feature_dim):
            report.fail(f"Runtime batch visual_features shape mismatch: {batch['visual_features'].shape}")
            return
        if batch["graph_adj"].shape != (batch_size, expected_total_nodes, expected_total_nodes):
            report.fail(f"Runtime batch graph_adj shape mismatch: {batch['graph_adj'].shape}")
            return
        if batch["graph_edge_types"].shape != (batch_size, expected_total_nodes, expected_total_nodes):
            report.fail(
                f"Runtime batch graph_edge_types shape mismatch: {batch['graph_edge_types'].shape}"
            )
            return
        if batch["graph_node_features"].shape != (batch_size, max_kg_nodes, d_kg):
            report.fail(
                f"Runtime batch graph_node_features shape mismatch: {batch['graph_node_features'].shape}"
            )
            return
        report.ok("Runtime batch shapes match dataset / model contract.")

        # Use the GQA-specific model on the runtime path so the validator
        # exercises the relation-aware GNN that production configs target.
        # `num_relations` must cover the relation vocab to match the trainer
        # contract; falling back to 0 here would silently disable relation
        # awareness and hide config / data mismatches.
        if relation_vocab_size <= 0:
            report.fail(
                "Runtime validation requires a non-empty relation vocab "
                "(gqa_relation_vocab.json must exist and pass schema check)."
            )
            return
        model = GQAVQAGNNModel(
            d_visual=feature_dim,
            d_kg=d_kg,
            num_answers=len(dataset.answer_to_idx),
            num_relations=relation_vocab_size,
            text_encoder_name=text_encoder,
            freeze_text_encoder=True,
        )
        model.eval()
        criterion = GQALoss()
        metric = GQAAccuracy(name="GQA_Accuracy")

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs["logits"]
            losses = criterion(**outputs, **batch)
            accuracy = metric(logits=logits, labels=batch["labels"])

        if logits.shape != (batch_size, len(dataset.answer_to_idx)):
            report.fail(f"Model logits shape mismatch: {logits.shape}")
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
        description="Strict validation for processed GQA artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--answer-vocab", default=None)
    parser.add_argument("--relation-vocab", default=None)
    parser.add_argument("--split", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--num-visual-nodes", type=int, default=100)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--d-kg", type=int, default=600)
    parser.add_argument("--max-kg-nodes", type=int, default=100)
    parser.add_argument("--num-answers", type=int, default=1842)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--text-encoder", default=_default_text_encoder())
    parser.add_argument("--runtime-batch-size", type=int, default=1)
    parser.add_argument("--skip-runtime-check", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    answer_vocab_path = Path(args.answer_vocab) if args.answer_vocab else data_dir / "gqa_answer_vocab.json"
    relation_vocab_path = (
        Path(args.relation_vocab) if args.relation_vocab else data_dir / "gqa_relation_vocab.json"
    )
    report = ValidationReport()

    answer_to_idx = validate_answer_vocab(answer_vocab_path, args.num_answers, report)
    relation_vocab = validate_relation_vocab(relation_vocab_path, report)
    relation_vocab_size = len(relation_vocab[1]) if relation_vocab is not None else 0

    for split in args.split:
        print(f"\n{'=' * 60}")
        print(f"Split: {split}")
        print("=" * 60)
        question_ids, image_ids = validate_questions(data_dir, split, report)
        validate_visual_features(
            data_dir=data_dir,
            split=split,
            expected_image_ids=image_ids,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
            sample_size=args.sample_size,
            report=report,
        )
        validate_visual_metadata(
            data_dir=data_dir,
            split=split,
            require_metadata=args.require_metadata,
            report=report,
        )
        validate_kg_graphs(
            data_dir=data_dir,
            split=split,
            expected_question_ids=question_ids,
            num_visual_nodes=args.num_visual_nodes,
            max_kg_nodes=args.max_kg_nodes,
            d_kg=args.d_kg,
            relation_vocab_size=relation_vocab_size,
            sample_size=args.sample_size,
            require_metadata=args.require_metadata,
            report=report,
        )

    if not args.skip_runtime_check and answer_to_idx is not None:
        validate_runtime_path(
            data_dir=data_dir,
            answer_vocab_path=answer_vocab_path,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
            d_kg=args.d_kg,
            max_kg_nodes=args.max_kg_nodes,
            text_encoder=args.text_encoder,
            runtime_batch_size=args.runtime_batch_size,
            relation_vocab_size=relation_vocab_size,
            report=report,
        )
    elif args.skip_runtime_check:
        report.warn("Runtime check skipped by user request.")

    success = report.summary()
    if not success:
        print("\nValidation FAILED.", file=sys.stderr)
        sys.exit(1)

    print("\nValidation passed. GQA data looks ready for runtime use.")


if __name__ == "__main__":
    main()
