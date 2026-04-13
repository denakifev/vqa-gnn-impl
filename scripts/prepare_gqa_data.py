"""
Paper-aligned GQA data pipeline for VQA-GNN.

This script is the single front door for the GQA preprocessing path:
  1. copy official balanced question files into the processed layout;
  2. build the fixed GQA answer space deterministically;
  3. package official GQA object features into runtime HDF5 files;
  4. build paper-like GQA graphs from official scene graphs;
  5. stage a Kaggle-ready full or mini package.

Processed runtime layout:

  data/gqa/
  ├── questions/
  │   ├── train_balanced_questions.json
  │   └── val_balanced_questions.json
  ├── visual_features/
  │   ├── train_features.h5
  │   └── val_features.h5
  ├── knowledge_graphs/
  │   ├── train_graphs.h5
  │   └── val_graphs.h5
  ├── metadata/
  │   ├── train_visual_features_metadata.json
  │   ├── val_visual_features_metadata.json
  │   ├── train_knowledge_graphs_metadata.json
  │   └── val_knowledge_graphs_metadata.json
  ├── gqa_answer_vocab.json
  └── gqa_relation_vocab.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

try:
    import h5py
except ImportError:
    h5py = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.gqa_preprocessing import (  # noqa: E402
    DEFAULT_GLOVE_DIM,
    GQAObjectFeatureStore,
    build_answer_vocab,
    build_question_graph,
    build_relation_vocab,
    collect_glove_token_vocab_from_scene_graphs,
    infer_split_name,
    load_gqa_questions,
    load_gqa_scene_graphs,
    load_glove_subset,
    save_answer_vocab,
    save_relation_vocab,
)


DEFAULT_TARGET_ANSWER_SIZE = 1842
DEFAULT_NUM_VISUAL_NODES = 100
DEFAULT_D_VISUAL = 2048
DEFAULT_GLOVE_DIMENSION = DEFAULT_GLOVE_DIM
DEFAULT_D_KG = DEFAULT_GLOVE_DIMENSION * 2
DEFAULT_MAX_KG_NODES = 100


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required for this command. Install it with `pip install h5py`.")


def _require_numpy() -> None:
    if np is None:
        raise ImportError("numpy is required for this command. Install it with `pip install numpy`.")


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _metadata_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "metadata"


def _visual_metadata_path(data_dir: str | Path, split: str) -> Path:
    return _metadata_dir(data_dir) / f"{split}_visual_features_metadata.json"


def _kg_metadata_path(data_dir: str | Path, split: str) -> Path:
    return _metadata_dir(data_dir) / f"{split}_knowledge_graphs_metadata.json"


def _package_manifest_path(data_dir: str | Path) -> Path:
    return _metadata_dir(data_dir) / "package_manifest.json"


def _relation_vocab_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "gqa_relation_vocab.json"


def _load_relation_vocab(path: str | Path) -> tuple[dict[str, int], list[str], dict]:
    with open(path) as f:
        payload = json.load(f)
    relation_to_idx = payload.get("relation_to_idx")
    idx_to_relation = payload.get("idx_to_relation")
    metadata = payload.get("metadata", {})
    if not isinstance(relation_to_idx, dict) or not isinstance(idx_to_relation, list):
        raise ValueError(f"Malformed relation vocab JSON: {path}")
    return relation_to_idx, idx_to_relation, metadata


def _question_image_ids(questions_json: str | Path) -> set[str]:
    questions = load_gqa_questions(questions_json)
    return {str(item["imageId"]) for item in questions.values()}


def _run_visual_conversion(
    raw_features: str | Path,
    info_json: str | Path,
    questions_json: str | Path,
    output_h5: str | Path,
    num_visual_nodes: int,
    feature_dim: int,
    compression: str,
) -> None:
    script_path = Path(__file__).with_name("prepare_visual_features.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--format",
        "gqa_h5",
        "--input",
        str(raw_features),
        "--info-json",
        str(info_json),
        "--questions-json",
        str(questions_json),
        "--output",
        str(output_h5),
        "--num-boxes",
        str(num_visual_nodes),
        "--feature-dim",
        str(feature_dim),
        "--compression",
        compression,
    ]
    subprocess.run(cmd, check=True)


def _summarize_visual_features(
    questions_json: str | Path,
    output_h5: str | Path,
    num_visual_nodes: int,
    feature_dim: int,
) -> dict:
    _require_h5py()
    questions = load_gqa_questions(questions_json)
    expected_image_ids = {str(item["imageId"]) for item in questions.values()}

    output_h5 = Path(output_h5)
    with h5py.File(output_h5, "a") as handle:
        image_ids = set(handle.keys())
        missing = sorted(expected_image_ids - image_ids)
        sample_key = next(iter(image_ids)) if image_ids else None
        sample_shape = None
        sample_dtype = None
        if sample_key is not None:
            sample = handle[sample_key][()]
            sample_shape = list(sample.shape)
            sample_dtype = str(sample.dtype)

        split = infer_split_name(questions_json)
        handle.attrs["dataset_name"] = "gqa"
        handle.attrs["split"] = split
        handle.attrs["num_boxes"] = num_visual_nodes
        handle.attrs["feature_dim"] = feature_dim
        handle.attrs["source_format"] = "official_gqa_objects_h5"
        handle.attrs["questions_json"] = str(questions_json)
        handle.attrs["image_count"] = len(image_ids)
        handle.attrs["fully_paper_aligned_format"] = True

    return {
        "split": infer_split_name(questions_json),
        "questions_json": str(questions_json),
        "output_h5": str(output_h5),
        "expected_unique_images": len(expected_image_ids),
        "written_image_entries": len(image_ids),
        "missing_image_ids": len(missing),
        "missing_image_ids_sample": missing[:10],
        "num_visual_nodes": num_visual_nodes,
        "feature_dim": feature_dim,
        "sample_shape": sample_shape,
        "sample_dtype": sample_dtype,
        "source_format": "official_gqa_objects_h5",
    }


def _answer_vocab_command(args: argparse.Namespace) -> None:
    answer_to_idx, idx_to_answer, metadata = build_answer_vocab(
        question_paths=args.questions,
        target_size=args.target_size,
    )
    if args.strict_target_size and len(idx_to_answer) != args.target_size:
        raise ValueError(
            f"Expected answer vocab size {args.target_size}, got {len(idx_to_answer)}. "
            "Use the official balanced train+val questions or rerun with --no-strict-target-size."
        )

    metadata["paper_target_alignment"] = (
        "exact_target_size" if len(idx_to_answer) == args.target_size else "size_mismatch"
    )
    save_answer_vocab(args.output_vocab, answer_to_idx, idx_to_answer, metadata)

    print(f"Saved answer vocab to {args.output_vocab}")
    print(f"  answers:      {len(idx_to_answer)}")
    print(f"  coverage_pct: {metadata['coverage_pct']}")
    print(f"  policy:       {metadata['selection_policy']}")


def _relation_vocab_command(args: argparse.Namespace) -> None:
    image_ids = set()
    for questions_json in args.questions:
        image_ids.update(_question_image_ids(questions_json))

    scene_graphs = load_gqa_scene_graphs(args.scene_graphs, image_ids=image_ids)
    missing = image_ids - set(scene_graphs.keys())
    if missing:
        raise ValueError(
            f"Missing {len(missing)} scene graphs referenced by the provided question files. "
            f"First 10: {sorted(missing)[:10]}"
        )

    relation_to_idx, idx_to_relation, metadata = build_relation_vocab(scene_graphs)
    metadata["questions"] = [str(Path(path)) for path in args.questions]
    metadata["scene_graphs"] = [str(Path(path)) for path in args.scene_graphs]
    save_relation_vocab(args.output_vocab, relation_to_idx, idx_to_relation, metadata)

    print(f"Saved relation vocab to {args.output_vocab}")
    print(f"  relations:    {len(idx_to_relation)}")
    print(f"  predicates:   {metadata['predicate_count']}")


def _visual_features_command(args: argparse.Namespace) -> None:
    _run_visual_conversion(
        raw_features=args.raw_features,
        info_json=args.info_json,
        questions_json=args.questions_json,
        output_h5=args.output,
        num_visual_nodes=args.num_visual_nodes,
        feature_dim=args.feature_dim,
        compression=args.compression,
    )

    metadata = _summarize_visual_features(
        questions_json=args.questions_json,
        output_h5=args.output,
        num_visual_nodes=args.num_visual_nodes,
        feature_dim=args.feature_dim,
    )
    metadata_path = _visual_metadata_path(Path(args.output).parents[1], metadata["split"])
    _write_json(metadata_path, metadata)

    print(f"Saved visual features to {args.output}")
    print(f"Saved visual metadata to {metadata_path}")
    if metadata["missing_image_ids"]:
        print(
            f"[WARN] Missing {metadata['missing_image_ids']} images referenced by questions. "
            "See metadata JSON for samples."
        )


def _build_knowledge_graphs(
    questions_json: str | Path,
    scene_graph_paths: list[str | Path],
    glove_path: str | Path,
    raw_features: str | Path,
    info_json: str | Path,
    relation_vocab_path: str | Path,
    output_h5: str | Path,
    metadata_path: str | Path,
    d_kg: int,
    glove_dim: int,
    num_visual_nodes: int,
    max_kg_nodes: int,
    compression: str,
    limit: int | None,
    scene_graphs: Optional[dict[str, dict]] = None,
    glove_embeddings: Optional[dict[str, np.ndarray]] = None,
    relation_to_idx: Optional[dict[str, int]] = None,
    idx_to_relation: Optional[list[str]] = None,
    feature_store: Optional[GQAObjectFeatureStore] = None,
) -> None:
    _require_h5py()
    _require_numpy()

    if d_kg != glove_dim * 2:
        raise ValueError(
            f"Requested d_kg={d_kg}, but paper-like textual SG nodes require 2 * glove_dim "
            f"({glove_dim * 2})."
        )

    questions = load_gqa_questions(questions_json)
    question_items = list(questions.items())
    if limit is not None:
        question_items = question_items[:limit]
    question_image_ids = {str(item["imageId"]) for _, item in question_items}

    if scene_graphs is None:
        scene_graphs = load_gqa_scene_graphs(scene_graph_paths, image_ids=question_image_ids)
    missing_scene_graphs = question_image_ids - set(scene_graphs.keys())
    if missing_scene_graphs:
        raise ValueError(
            f"Missing {len(missing_scene_graphs)} scene graphs referenced by {questions_json}. "
            f"First 10: {sorted(missing_scene_graphs)[:10]}"
        )

    if relation_to_idx is None or idx_to_relation is None:
        relation_to_idx, idx_to_relation, _ = _load_relation_vocab(relation_vocab_path)

    if glove_embeddings is None:
        required_tokens = collect_glove_token_vocab_from_scene_graphs(scene_graphs)
        glove_embeddings = load_glove_subset(glove_path, required_tokens, target_dim=glove_dim)

    owns_feature_store = feature_store is None
    if feature_store is None:
        feature_store = GQAObjectFeatureStore(raw_features, info_json)

    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    compression_value = None if compression == "none" else compression

    token_vocab = collect_glove_token_vocab_from_scene_graphs(scene_graphs)
    entity_counter: Counter[str] = Counter()
    cached_visual_info: dict[str, tuple[np.ndarray, int]] = {}
    stats = {
        "split": infer_split_name(questions_json),
        "questions_json": str(questions_json),
        "scene_graph_paths": [str(Path(path)) for path in scene_graph_paths],
        "glove_path": str(glove_path),
        "raw_features": str(raw_features),
        "info_json": str(info_json),
        "relation_vocab_path": str(relation_vocab_path),
        "output_h5": str(output_h5),
        "d_kg": d_kg,
        "glove_dim": glove_dim,
        "num_visual_nodes": num_visual_nodes,
        "max_kg_nodes": max_kg_nodes,
        "total_questions": len(question_items),
        "total_unique_images": len(question_image_ids),
        "questions_with_linked_entities": 0,
        "questions_without_linked_entities": 0,
        "questions_with_multiword_entity": 0,
        "questions_with_textual_truncation": 0,
        "total_linked_entities": 0,
        "total_textual_nodes": 0,
        "total_visual_nodes": 0,
        "total_aligned_visual_nodes": 0,
        "total_visual_relation_edges": 0,
        "total_textual_relation_edges": 0,
        "max_textual_nodes": 0,
        "max_visual_relation_edges": 0,
        "max_textual_relation_edges": 0,
        "no_match_question_ids_sample": [],
        "fully_connected_fallback_used": False,
        "graph_mode": "question_context_plus_visual_textual_scene_graphs",
        "textual_node_features": "concat(glove(object_name), glove(mean_attributes))",
        "visual_alignment": "bbox_iou_greedy",
        "bbox_iou_threshold": 0.5,
        "conceptnet_used": False,
    }

    try:
        with h5py.File(output_h5, "w") as handle:
            for question_id, item in tqdm(question_items, desc="Building GQA scene graphs"):
                image_id = str(item["imageId"])
                if image_id not in cached_visual_info:
                    cached_visual_info[image_id] = feature_store.get_boxes(image_id)
                visual_boxes, actual_visual_nodes = cached_visual_info[image_id]

                result = build_question_graph(
                    question=item["question"],
                    scene_graph=scene_graphs[image_id],
                    glove_embeddings=glove_embeddings,
                    relation_to_idx=relation_to_idx,
                    max_kg_nodes=max_kg_nodes,
                    num_visual_nodes=num_visual_nodes,
                    glove_dim=glove_dim,
                    visual_boxes=visual_boxes,
                    actual_visual_nodes=actual_visual_nodes,
                )

                group = handle.create_group(str(question_id))
                group.create_dataset(
                    "node_features",
                    data=result.node_features.astype(np.float32),
                    compression=compression_value,
                )
                group.create_dataset(
                    "adj_matrix",
                    data=result.adj_matrix.astype(np.float32),
                    compression=compression_value,
                )
                group.create_dataset(
                    "graph_edge_types",
                    data=result.edge_types.astype(np.int32),
                    compression=compression_value,
                )
                group.create_dataset(
                    "node_types",
                    data=result.node_types.astype(np.int32),
                    compression=compression_value,
                )

                n_linked = len(result.linked_entities)
                n_textual = len(result.selected_object_ids)

                stats["total_linked_entities"] += n_linked
                stats["total_textual_nodes"] += n_textual
                stats["total_visual_nodes"] += result.actual_visual_nodes
                stats["total_aligned_visual_nodes"] += result.aligned_visual_nodes
                stats["total_visual_relation_edges"] += result.visual_relation_edges
                stats["total_textual_relation_edges"] += result.textual_relation_edges
                stats["max_textual_nodes"] = max(stats["max_textual_nodes"], n_textual)
                stats["max_visual_relation_edges"] = max(
                    stats["max_visual_relation_edges"], result.visual_relation_edges
                )
                stats["max_textual_relation_edges"] = max(
                    stats["max_textual_relation_edges"], result.textual_relation_edges
                )

                if n_linked:
                    stats["questions_with_linked_entities"] += 1
                else:
                    stats["questions_without_linked_entities"] += 1
                    if len(stats["no_match_question_ids_sample"]) < 20:
                        stats["no_match_question_ids_sample"].append(str(question_id))

                if any(entity.is_multiword for entity in result.linked_entities):
                    stats["questions_with_multiword_entity"] += 1
                if result.truncated_objects:
                    stats["questions_with_textual_truncation"] += 1

                for entity in result.linked_entities:
                    entity_counter[entity.canonical] += 1

            handle.attrs["dataset_name"] = "gqa"
            handle.attrs["split"] = stats["split"]
            handle.attrs["d_kg"] = d_kg
            handle.attrs["glove_dim"] = glove_dim
            handle.attrs["num_visual_nodes"] = num_visual_nodes
            handle.attrs["max_kg_nodes"] = max_kg_nodes
            handle.attrs["question_count"] = len(question_items)
            handle.attrs["graph_mode"] = "official_scene_graph"
            handle.attrs["scene_graph_paths"] = json.dumps(stats["scene_graph_paths"])
            handle.attrs["glove_path"] = str(glove_path)
            handle.attrs["relation_vocab_path"] = str(relation_vocab_path)
            handle.attrs["conceptnet_used"] = False
            handle.attrs["fully_connected_fallback_used"] = False
            handle.attrs["graph_edge_type_count"] = len(idx_to_relation)

        total_questions = max(stats["total_questions"], 1)
        stats["questions_with_linked_entities_pct"] = round(
            100.0 * stats["questions_with_linked_entities"] / total_questions, 6
        )
        stats["questions_without_linked_entities_pct"] = round(
            100.0 * stats["questions_without_linked_entities"] / total_questions, 6
        )
        stats["questions_with_multiword_entity_pct"] = round(
            100.0 * stats["questions_with_multiword_entity"] / total_questions, 6
        )
        stats["questions_with_textual_truncation_pct"] = round(
            100.0 * stats["questions_with_textual_truncation"] / total_questions, 6
        )
        stats["avg_linked_entities"] = round(stats["total_linked_entities"] / total_questions, 6)
        stats["avg_textual_nodes"] = round(stats["total_textual_nodes"] / total_questions, 6)
        stats["avg_visual_nodes"] = round(stats["total_visual_nodes"] / total_questions, 6)
        stats["avg_aligned_visual_nodes"] = round(
            stats["total_aligned_visual_nodes"] / total_questions, 6
        )
        stats["avg_visual_relation_edges"] = round(
            stats["total_visual_relation_edges"] / total_questions, 6
        )
        stats["avg_textual_relation_edges"] = round(
            stats["total_textual_relation_edges"] / total_questions, 6
        )
        stats["avg_graph_size_total_nodes"] = round(
            num_visual_nodes + 1 + stats["avg_textual_nodes"], 6
        )
        stats["glove_token_vocab_size"] = len(token_vocab)
        stats["glove_loaded_token_count"] = len(glove_embeddings)
        stats["glove_token_coverage_pct"] = round(
            100.0 * len(glove_embeddings) / max(len(token_vocab), 1), 6
        )
        stats["top_linked_entities"] = entity_counter.most_common(50)

        _write_json(metadata_path, stats)
    finally:
        if owns_feature_store:
            feature_store.close()


def _knowledge_graphs_command(args: argparse.Namespace) -> None:
    split = infer_split_name(args.questions_json)
    metadata_path = args.metadata_output or _kg_metadata_path(Path(args.output).parents[1], split)
    relation_to_idx, idx_to_relation, _ = _load_relation_vocab(args.relation_vocab)
    image_ids = _question_image_ids(args.questions_json)
    scene_graphs = load_gqa_scene_graphs(args.scene_graphs, image_ids=image_ids)
    required_tokens = collect_glove_token_vocab_from_scene_graphs(scene_graphs)
    glove_embeddings = load_glove_subset(args.glove, required_tokens, target_dim=args.glove_dim)
    with GQAObjectFeatureStore(args.raw_features, args.info_json) as feature_store:
        _build_knowledge_graphs(
            questions_json=args.questions_json,
            scene_graph_paths=args.scene_graphs,
            glove_path=args.glove,
            raw_features=args.raw_features,
            info_json=args.info_json,
            relation_vocab_path=args.relation_vocab,
            output_h5=args.output,
            metadata_path=metadata_path,
            d_kg=args.d_kg,
            glove_dim=args.glove_dim,
            num_visual_nodes=args.num_visual_nodes,
            max_kg_nodes=args.max_kg_nodes,
            compression=args.compression,
            limit=args.limit,
            scene_graphs=scene_graphs,
            glove_embeddings=glove_embeddings,
            relation_to_idx=relation_to_idx,
            idx_to_relation=idx_to_relation,
            feature_store=feature_store,
        )

    print(f"Saved knowledge graphs to {args.output}")
    print(f"Saved KG metadata to {metadata_path}")


def _copy_official_questions(
    train_questions: str | Path,
    val_questions: str | Path,
    processed_dir: str | Path,
) -> tuple[Path, Path]:
    processed_dir = Path(processed_dir)
    questions_dir = processed_dir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    train_dst = questions_dir / "train_balanced_questions.json"
    val_dst = questions_dir / "val_balanced_questions.json"
    _copy_file(train_questions, train_dst)
    _copy_file(val_questions, val_dst)
    return train_dst, val_dst


def _prepare_all_command(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _metadata_dir(processed_dir).mkdir(parents=True, exist_ok=True)

    train_questions, val_questions = _copy_official_questions(
        train_questions=args.train_questions,
        val_questions=args.val_questions,
        processed_dir=processed_dir,
    )

    vocab_output = processed_dir / "gqa_answer_vocab.json"
    answer_to_idx, idx_to_answer, metadata = build_answer_vocab(
        question_paths=[train_questions, val_questions],
        target_size=args.target_size,
    )
    if args.strict_target_size and len(idx_to_answer) != args.target_size:
        raise ValueError(
            f"Expected answer vocab size {args.target_size}, got {len(idx_to_answer)}."
        )
    metadata["paper_target_alignment"] = (
        "exact_target_size" if len(idx_to_answer) == args.target_size else "size_mismatch"
    )
    save_answer_vocab(vocab_output, answer_to_idx, idx_to_answer, metadata)

    visual_dir = processed_dir / "visual_features"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for split, questions_json in (("train", train_questions), ("val", val_questions)):
        visual_output = visual_dir / f"{split}_features.h5"
        _run_visual_conversion(
            raw_features=args.raw_features,
            info_json=args.info_json,
            questions_json=questions_json,
            output_h5=visual_output,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
            compression=args.compression,
        )
        visual_metadata = _summarize_visual_features(
            questions_json=questions_json,
            output_h5=visual_output,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=args.feature_dim,
        )
        _write_json(_visual_metadata_path(processed_dir, split), visual_metadata)

    all_image_ids = _question_image_ids(train_questions) | _question_image_ids(val_questions)
    scene_graphs = load_gqa_scene_graphs(args.scene_graphs, image_ids=all_image_ids)
    missing_scene_graphs = all_image_ids - set(scene_graphs.keys())
    if missing_scene_graphs:
        raise ValueError(
            f"Missing {len(missing_scene_graphs)} scene graphs referenced by train/val questions. "
            f"First 10: {sorted(missing_scene_graphs)[:10]}"
        )

    relation_vocab_output = _relation_vocab_path(processed_dir)
    relation_to_idx, idx_to_relation, relation_metadata = build_relation_vocab(scene_graphs)
    relation_metadata["questions"] = [str(train_questions), str(val_questions)]
    relation_metadata["scene_graphs"] = [str(Path(path)) for path in args.scene_graphs]
    save_relation_vocab(
        relation_vocab_output,
        relation_to_idx=relation_to_idx,
        idx_to_relation=idx_to_relation,
        metadata=relation_metadata,
    )

    glove_tokens = collect_glove_token_vocab_from_scene_graphs(scene_graphs)
    glove_embeddings = load_glove_subset(args.glove, glove_tokens, target_dim=args.glove_dim)

    kg_dir = processed_dir / "knowledge_graphs"
    kg_dir.mkdir(parents=True, exist_ok=True)
    with GQAObjectFeatureStore(args.raw_features, args.info_json) as feature_store:
        for split, questions_json in (("train", train_questions), ("val", val_questions)):
            split_image_ids = _question_image_ids(questions_json)
            split_scene_graphs = {
                image_id: scene_graphs[image_id] for image_id in split_image_ids if image_id in scene_graphs
            }
            _build_knowledge_graphs(
                questions_json=questions_json,
                scene_graph_paths=args.scene_graphs,
                glove_path=args.glove,
                raw_features=args.raw_features,
                info_json=args.info_json,
                relation_vocab_path=relation_vocab_output,
                output_h5=kg_dir / f"{split}_graphs.h5",
                metadata_path=_kg_metadata_path(processed_dir, split),
                d_kg=args.d_kg,
                glove_dim=args.glove_dim,
                num_visual_nodes=args.num_visual_nodes,
                max_kg_nodes=args.max_kg_nodes,
                compression=args.compression,
                limit=args.limit,
                scene_graphs=split_scene_graphs,
                glove_embeddings=glove_embeddings,
                relation_to_idx=relation_to_idx,
                idx_to_relation=idx_to_relation,
                feature_store=feature_store,
            )

    manifest = {
        "dataset_name": "gqa",
        "processed_dir": str(processed_dir),
        "question_files": {
            "train": str(train_questions),
            "val": str(val_questions),
        },
        "answer_vocab_path": str(vocab_output),
        "relation_vocab_path": str(relation_vocab_output),
        "visual_features": {
            "train": str(visual_dir / "train_features.h5"),
            "val": str(visual_dir / "val_features.h5"),
        },
        "knowledge_graphs": {
            "train": str(kg_dir / "train_graphs.h5"),
            "val": str(kg_dir / "val_graphs.h5"),
        },
        "num_visual_nodes": args.num_visual_nodes,
        "feature_dim": args.feature_dim,
        "glove_dim": args.glove_dim,
        "d_kg": args.d_kg,
        "max_kg_nodes": args.max_kg_nodes,
        "paper_target_answer_size": args.target_size,
        "graph_mode": "official_scene_graph",
        "conceptnet_used": False,
        "fully_connected_fallback_used": False,
    }
    _write_json(_package_manifest_path(processed_dir), manifest)

    print(f"Prepared processed GQA dataset under {processed_dir}")
    print("Next step:")
    print(
        f"  python scripts/validate_gqa_data.py --data-dir {processed_dir} "
        f"--answer-vocab {vocab_output}"
    )


def _subset_questions(src: str | Path, dst: str | Path, limit: int) -> dict[str, dict]:
    questions = load_gqa_questions(src)
    subset = dict(list(questions.items())[:limit])
    _write_json(dst, subset)
    return subset


def _subset_visual_h5(src_h5: str | Path, dst_h5: str | Path, image_ids: set[str]) -> dict:
    _require_h5py()
    dst_h5 = Path(dst_h5)
    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for image_id in tqdm(sorted(image_ids), desc=f"Subsetting visual {dst_h5.stem}"):
            if image_id not in src:
                continue
            dst.create_dataset(image_id, data=src[image_id][()], compression="gzip")
            written += 1
    return {"expected_images": len(image_ids), "written_images": written}


def _subset_graph_h5(src_h5: str | Path, dst_h5: str | Path, question_ids: set[str]) -> dict:
    _require_h5py()
    dst_h5 = Path(dst_h5)
    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for question_id in tqdm(sorted(question_ids), desc=f"Subsetting KG {dst_h5.stem}"):
            if question_id not in src:
                continue
            group = dst.create_group(question_id)
            for dataset_name in ("node_features", "adj_matrix", "graph_edge_types", "node_types"):
                if dataset_name not in src[question_id]:
                    continue
                group.create_dataset(
                    dataset_name,
                    data=src[question_id][dataset_name][()],
                    compression="gzip",
                )
            written += 1
    return {"expected_graphs": len(question_ids), "written_graphs": written}


def _full_required_artifacts(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return [
        data_dir / "questions" / "train_balanced_questions.json",
        data_dir / "questions" / "val_balanced_questions.json",
        data_dir / "visual_features" / "train_features.h5",
        data_dir / "visual_features" / "val_features.h5",
        data_dir / "knowledge_graphs" / "train_graphs.h5",
        data_dir / "knowledge_graphs" / "val_graphs.h5",
        data_dir / "gqa_answer_vocab.json",
        data_dir / "gqa_relation_vocab.json",
    ]


def _copy_optional_metadata(src_dir: str | Path, dst_dir: str | Path) -> None:
    src_meta = _metadata_dir(src_dir)
    dst_meta = _metadata_dir(dst_dir)
    if not src_meta.exists():
        return
    dst_meta.mkdir(parents=True, exist_ok=True)
    for item in src_meta.glob("*.json"):
        _copy_file(item, dst_meta / item.name)


def _stage_command(args: argparse.Namespace) -> None:
    _require_h5py()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    missing = [path for path in _full_required_artifacts(data_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed GQA artifacts:\n" + "\n".join(f"  - {path}" for path in missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "questions").mkdir(exist_ok=True)
    (output_dir / "visual_features").mkdir(exist_ok=True)
    (output_dir / "knowledge_graphs").mkdir(exist_ok=True)
    _metadata_dir(output_dir).mkdir(exist_ok=True)

    question_stats: dict[str, dict] = {}
    if args.variant == "mini":
        for split in ("train", "val"):
            subset = _subset_questions(
                src=data_dir / "questions" / f"{split}_balanced_questions.json",
                dst=output_dir / "questions" / f"{split}_balanced_questions.json",
                limit=args.mini_n,
            )
            question_ids = set(subset.keys())
            image_ids = {str(item["imageId"]) for item in subset.values()}
            question_stats[split] = {
                "question_count": len(question_ids),
                "image_count": len(image_ids),
            }

            visual_stats = _subset_visual_h5(
                src_h5=data_dir / "visual_features" / f"{split}_features.h5",
                dst_h5=output_dir / "visual_features" / f"{split}_features.h5",
                image_ids=image_ids,
            )
            graph_stats = _subset_graph_h5(
                src_h5=data_dir / "knowledge_graphs" / f"{split}_graphs.h5",
                dst_h5=output_dir / "knowledge_graphs" / f"{split}_graphs.h5",
                question_ids=question_ids,
            )
            question_stats[split].update(visual_stats)
            question_stats[split].update(graph_stats)
    else:
        for artifact in _full_required_artifacts(data_dir):
            relative = artifact.relative_to(data_dir)
            _copy_file(artifact, output_dir / relative)

        for split in ("train", "val"):
            questions = load_gqa_questions(data_dir / "questions" / f"{split}_balanced_questions.json")
            question_stats[split] = {
                "question_count": len(questions),
                "image_count": len({str(item["imageId"]) for item in questions.values()}),
            }

    _copy_optional_metadata(data_dir, output_dir)
    _copy_file(data_dir / "gqa_answer_vocab.json", output_dir / "gqa_answer_vocab.json")
    _copy_file(data_dir / "gqa_relation_vocab.json", output_dir / "gqa_relation_vocab.json")

    dataset_id = args.dataset_id or (
        "gqa-gnn-data-mini" if args.variant == "mini" else "gqa-gnn-data"
    )
    dataset_title = args.dataset_title or (
        f"GQA VQA-GNN data (mini {args.mini_n} per split)"
        if args.variant == "mini"
        else "GQA VQA-GNN data (full balanced split)"
    )

    _write_json(
        output_dir / "dataset-metadata.json",
        {
            "title": dataset_title,
            "id": f"KAGGLE_USERNAME/{dataset_id}",
            "licenses": [{"name": "CC BY 4.0"}],
        },
    )

    manifest = {
        "dataset_name": "gqa",
        "variant": args.variant,
        "source_data_dir": str(data_dir),
        "staged_output_dir": str(output_dir),
        "mini_n": args.mini_n if args.variant == "mini" else None,
        "question_stats": question_stats,
        "metadata_source": (
            "copied_from_processed_source_splits" if args.variant == "mini" else "staged_split_exact"
        ),
        "paths": {
            "answer_vocab": "gqa_answer_vocab.json",
            "relation_vocab": "gqa_relation_vocab.json",
            "questions": {
                "train": "questions/train_balanced_questions.json",
                "val": "questions/val_balanced_questions.json",
            },
            "visual_features": {
                "train": "visual_features/train_features.h5",
                "val": "visual_features/val_features.h5",
            },
            "knowledge_graphs": {
                "train": "knowledge_graphs/train_graphs.h5",
                "val": "knowledge_graphs/val_graphs.h5",
            },
        },
        "paper_target_answer_size": DEFAULT_TARGET_ANSWER_SIZE,
        "graph_mode": "official_scene_graph",
        "kaggle_username_placeholder": "KAGGLE_USERNAME",
    }
    _write_json(_package_manifest_path(output_dir), manifest)

    print(f"Staged GQA package to {output_dir}")
    print("Next step:")
    print(
        f"  python scripts/validate_gqa_data.py --data-dir {output_dir} "
        f"--answer-vocab {output_dir / 'gqa_answer_vocab.json'} --skip-runtime-check"
    )


def _legacy_parser(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy compatibility mode for prepare_gqa_data.py")
    parser.add_argument("--build-vocab", action="store_true")
    parser.add_argument("--build-graphs", action="store_true")
    parser.add_argument("--validate-visual", action="store_true")
    parser.add_argument("--questions", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--output-vocab", default="data/gqa/gqa_answer_vocab.json")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TARGET_ANSWER_SIZE)
    parser.add_argument("--num-visual-nodes", type=int, default=DEFAULT_NUM_VISUAL_NODES)
    parser.add_argument("--visual-h5")
    parser.add_argument("--strict-target-size", action="store_true", default=True)
    parser.add_argument("--no-strict-target-size", action="store_false", dest="strict_target_size")
    return parser.parse_args(argv)


def _run_legacy_mode(argv: list[str]) -> bool:
    legacy_flags = {"--build-vocab", "--build-graphs", "--validate-visual"}
    if not any(flag in argv for flag in legacy_flags):
        return False

    args = _legacy_parser(argv)

    if args.build_vocab:
        _answer_vocab_command(
            argparse.Namespace(
                questions=args.questions,
                output_vocab=args.output_vocab,
                target_size=args.top_n,
                strict_target_size=args.strict_target_size,
            )
        )

    if args.build_graphs:
        raise ValueError(
            "Legacy --build-graphs mode was removed for GQA. "
            "Use `relation-vocab` + `knowledge-graphs` with official scene graphs instead."
        )

    if args.validate_visual:
        if not args.questions or len(args.questions) != 1 or not args.visual_h5:
            raise ValueError("--validate-visual expects one --questions path and --visual-h5.")
        metadata = _summarize_visual_features(
            questions_json=args.questions[0],
            output_h5=args.visual_h5,
            num_visual_nodes=args.num_visual_nodes,
            feature_dim=DEFAULT_D_VISUAL,
        )
        print(json.dumps(metadata, indent=2))

    print("[INFO] Legacy flag mode executed on top of the new GQA pipeline.")
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-aligned GQA data pipeline for VQA-GNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    answer_vocab = subparsers.add_parser("answer-vocab", help="Build deterministic GQA answer vocab.")
    answer_vocab.add_argument("--questions", nargs="+", required=True)
    answer_vocab.add_argument("--output-vocab", required=True)
    answer_vocab.add_argument("--target-size", type=int, default=DEFAULT_TARGET_ANSWER_SIZE)
    answer_vocab.add_argument("--strict-target-size", action="store_true", default=True)
    answer_vocab.add_argument("--no-strict-target-size", action="store_false", dest="strict_target_size")
    answer_vocab.set_defaults(handler=_answer_vocab_command)

    relation_vocab = subparsers.add_parser(
        "relation-vocab",
        help="Build a deterministic GQA relation vocabulary from official scene graphs.",
    )
    relation_vocab.add_argument("--questions", nargs="+", required=True)
    relation_vocab.add_argument("--scene-graphs", nargs="+", required=True)
    relation_vocab.add_argument("--output-vocab", required=True)
    relation_vocab.set_defaults(handler=_relation_vocab_command)

    visual = subparsers.add_parser(
        "visual-features", help="Package official GQA object features into runtime HDF5."
    )
    visual.add_argument("--raw-features", required=True)
    visual.add_argument("--info-json", required=True)
    visual.add_argument("--questions-json", required=True)
    visual.add_argument("--output", required=True)
    visual.add_argument("--num-visual-nodes", type=int, default=DEFAULT_NUM_VISUAL_NODES)
    visual.add_argument("--feature-dim", type=int, default=DEFAULT_D_VISUAL)
    visual.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    visual.set_defaults(handler=_visual_features_command)

    graphs = subparsers.add_parser(
        "knowledge-graphs",
        help="Build paper-like GQA graphs from official scene graphs.",
    )
    graphs.add_argument("--questions-json", required=True)
    graphs.add_argument("--scene-graphs", nargs="+", required=True)
    graphs.add_argument("--glove", required=True)
    graphs.add_argument("--raw-features", required=True)
    graphs.add_argument("--info-json", required=True)
    graphs.add_argument("--relation-vocab", required=True)
    graphs.add_argument("--output", required=True)
    graphs.add_argument("--metadata-output")
    graphs.add_argument("--glove-dim", type=int, default=DEFAULT_GLOVE_DIMENSION)
    graphs.add_argument("--d-kg", type=int, default=DEFAULT_D_KG)
    graphs.add_argument("--num-visual-nodes", type=int, default=DEFAULT_NUM_VISUAL_NODES)
    graphs.add_argument("--max-kg-nodes", type=int, default=DEFAULT_MAX_KG_NODES)
    graphs.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    graphs.add_argument("--limit", type=int, default=None)
    graphs.set_defaults(handler=_knowledge_graphs_command)

    prepare_all = subparsers.add_parser("prepare-all", help="Run the full raw -> processed GQA pipeline.")
    prepare_all.add_argument("--train-questions", required=True)
    prepare_all.add_argument("--val-questions", required=True)
    prepare_all.add_argument("--scene-graphs", nargs="+", required=True)
    prepare_all.add_argument("--glove", required=True)
    prepare_all.add_argument("--raw-features", required=True)
    prepare_all.add_argument("--info-json", required=True)
    prepare_all.add_argument("--processed-dir", required=True)
    prepare_all.add_argument("--target-size", type=int, default=DEFAULT_TARGET_ANSWER_SIZE)
    prepare_all.add_argument("--strict-target-size", action="store_true", default=True)
    prepare_all.add_argument("--no-strict-target-size", action="store_false", dest="strict_target_size")
    prepare_all.add_argument("--num-visual-nodes", type=int, default=DEFAULT_NUM_VISUAL_NODES)
    prepare_all.add_argument("--feature-dim", type=int, default=DEFAULT_D_VISUAL)
    prepare_all.add_argument("--glove-dim", type=int, default=DEFAULT_GLOVE_DIMENSION)
    prepare_all.add_argument("--d-kg", type=int, default=DEFAULT_D_KG)
    prepare_all.add_argument("--max-kg-nodes", type=int, default=DEFAULT_MAX_KG_NODES)
    prepare_all.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    prepare_all.add_argument("--limit", type=int, default=None)
    prepare_all.set_defaults(handler=_prepare_all_command)

    stage = subparsers.add_parser("stage", help="Create a Kaggle-ready full or mini GQA package.")
    stage.add_argument("--data-dir", required=True)
    stage.add_argument("--output-dir", required=True)
    stage.add_argument("--variant", choices=["full", "mini"], default="full")
    stage.add_argument("--mini-n", type=int, default=500)
    stage.add_argument("--dataset-id")
    stage.add_argument("--dataset-title")
    stage.set_defaults(handler=_stage_command)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    argv_list = list(argv if argv is not None else sys.argv[1:])
    if _run_legacy_mode(argv_list):
        return

    parser = _build_parser()
    args = parser.parse_args(argv_list)
    args.handler(args)


if __name__ == "__main__":
    main()
