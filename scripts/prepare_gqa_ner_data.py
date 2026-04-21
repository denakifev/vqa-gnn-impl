"""
Prepare a weakly-supervised GQA NER dataset from official GQA resources.

This dataset is intended for entity extraction / linking experiments over GQA
questions. It supports two deterministic supervision sources:

  1. question_semantics
     Uses official balanced-question annotations and semantic programs.
     This path is available from train/val_balanced_questions.json alone.

  2. scene_graph_alignment
     Uses official scene graphs to align object names / attributes / relations
     back onto question spans.

Labels are derived as:

  - OBJECT    question span matching an object reference
  - ATTRIBUTE question span matching an attribute-like phrase
  - RELATION  question span matching a relation phrase

Output layout:

  data/gqa_ner/
  ├── train.jsonl
  ├── val.jsonl
  ├── label_schema.json
  └── metadata/
      ├── train_metadata.json
      ├── val_metadata.json
      └── package_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.gqa_preprocessing import (  # noqa: E402
    infer_split_name,
    link_question_entities,
    load_gqa_questions,
    load_gqa_scene_graphs,
    parse_scene_graph_objects,
    tokenize_for_linking,
)


ENTITY_TYPE_OBJECT = "OBJECT"
ENTITY_TYPE_ATTRIBUTE = "ATTRIBUTE"
ENTITY_TYPE_RELATION = "RELATION"

ENTITY_TYPES = [
    ENTITY_TYPE_OBJECT,
    ENTITY_TYPE_ATTRIBUTE,
    ENTITY_TYPE_RELATION,
]
LABELS = [
    "O",
    "B-OBJECT",
    "I-OBJECT",
    "B-ATTRIBUTE",
    "I-ATTRIBUTE",
    "B-RELATION",
    "I-RELATION",
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
TYPE_PRIORITY = {
    ENTITY_TYPE_OBJECT: 0,
    ENTITY_TYPE_ATTRIBUTE: 1,
    ENTITY_TYPE_RELATION: 2,
}
ENTITY_PRIORITY = {
    "question_annotation_object": 0,
    "semantic_object": 1,
    "semantic_relation": 2,
    "semantic_attribute": 3,
}
ARTICLES = {"a", "an", "the"}
SEMANTIC_SUPERVISION = "question_semantics"
SCENE_GRAPH_SUPERVISION = "scene_graph_alignment"
_SEMANTIC_ID_RE = re.compile(r"\s*\(([^()]*)\)\s*$")
_NOT_EXPR_RE = re.compile(r"^not\((.+)\)$")


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _metadata_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "metadata"


def _split_metadata_path(data_dir: str | Path, split: str) -> Path:
    return _metadata_dir(data_dir) / f"{split}_metadata.json"


def _package_manifest_path(data_dir: str | Path) -> Path:
    return _metadata_dir(data_dir) / "package_manifest.json"


def _label_schema_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "label_schema.json"


def _resolve_kaggle_username(explicit: str | None) -> str | None:
    if explicit:
        return explicit

    env_value = os.environ.get("KAGGLE_USERNAME")
    if env_value:
        return env_value

    credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    if credentials_path.exists():
        with open(credentials_path) as f:
            payload = json.load(f)
        username = payload.get("username")
        if username:
            return str(username)
    return None


def _build_scene_graph_lexicon(scene_graph: dict) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, list[dict]]]:
    objects = parse_scene_graph_objects(scene_graph)
    object_names: dict[str, set[str]] = defaultdict(set)
    attributes: dict[str, set[str]] = defaultdict(set)
    relations: dict[str, list[dict]] = defaultdict(list)
    object_name_by_id = {obj.object_id: obj.name for obj in objects}

    for obj in objects:
        object_names[obj.name].add(obj.object_id)
        for attr in obj.attributes:
            attributes[attr].add(obj.object_id)
        for predicate, target_id in obj.relations:
            relations[predicate].append(
                {
                    "source_object_id": obj.object_id,
                    "target_object_id": target_id,
                    "source_name": obj.name,
                    "target_name": object_name_by_id.get(target_id),
                }
            )

    return object_names, attributes, relations


def _tokenize_question_for_annotations(text: str) -> list[str]:
    tokens: list[str] = []
    for token in text.strip().split():
        normalized = token.casefold().strip(".,?!;:'\"()[]{}")
        if normalized:
            tokens.append(normalized)
    return tokens


def _parse_span_key(span_key: str) -> tuple[int, int]:
    if ":" in span_key:
        start_str, end_str = span_key.split(":", 1)
        return int(start_str), int(end_str)
    start = int(span_key)
    return start, start + 1


def _normalize_phrase_choices(raw_phrase: str) -> list[str]:
    phrase = raw_phrase.strip()
    if not phrase or phrase == "_":
        return []

    choices: list[str] = []
    for item in phrase.split("|"):
        candidate = item.strip().strip("_")
        if not candidate:
            continue
        not_match = _NOT_EXPR_RE.match(candidate)
        if not_match:
            inner = not_match.group(1).strip()
            if inner:
                choices.append(f"not {inner}")
                choices.append(inner)
            continue
        choices.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for choice in choices:
        normalized = " ".join(tokenize_for_linking(choice))
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _find_phrase_spans(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    phrase_tokens = phrase.split()
    if not phrase_tokens or len(phrase_tokens) > len(tokens):
        return []

    spans: list[tuple[int, int]] = []
    width = len(phrase_tokens)
    for start in range(len(tokens) - width + 1):
        if tokens[start : start + width] == phrase_tokens:
            spans.append((start, start + width))
    return spans


def _semantic_argument_name(argument: str) -> tuple[str | None, str | None]:
    match = _SEMANTIC_ID_RE.search(argument.strip())
    object_id = match.group(1).strip() if match else None
    prefix = argument[: match.start()] if match else argument
    normalized_name = " ".join(tokenize_for_linking(prefix))
    return (normalized_name or None), object_id


def _semantic_relation_payload(argument: str) -> tuple[list[str], str | None, str | None]:
    match = _SEMANTIC_ID_RE.search(argument.strip())
    object_id = match.group(1).strip() if match else None
    prefix = argument[: match.start()] if match else argument
    parts = [part.strip() for part in prefix.split(",")]
    if len(parts) < 3:
        return [], None, object_id
    object_name = " ".join(tokenize_for_linking(parts[0])) if parts[0] and parts[0] != "_" else None
    relation_choices = _normalize_phrase_choices(parts[1])
    return relation_choices, object_name, object_id


def _attribute_choices_from_step(step: dict) -> list[str]:
    operation = (step.get("operation") or "").strip().lower()
    argument = str(step.get("argument") or "").strip()
    if not argument:
        return []

    if operation in {"select", "query", "relate", "verify rel", "choose rel", "exist", "and", "or"}:
        return []
    if operation == "choose name":
        return []
    return _normalize_phrase_choices(argument)


def _build_semantic_candidates(tokens: list[str], item: dict) -> list[dict]:
    entities: list[dict] = []
    annotations = item.get("annotations", {})
    question_annotations = annotations.get("question") or {}

    for span_key, raw_object_id in question_annotations.items():
        start, end = _parse_span_key(str(span_key))
        if start < 0 or end > len(tokens) or start >= end:
            continue
        entities.append(
            {
                "label": ENTITY_TYPE_OBJECT,
                "canonical": " ".join(tokens[start:end]),
                "surface": " ".join(tokens[start:end]),
                "start": start,
                "end": end,
                "match_type": "question_annotation",
                "source_priority": ENTITY_PRIORITY["question_annotation_object"],
                "object_ids": [str(raw_object_id)],
            }
        )

    semantic_steps = item.get("semantic") or []
    for step in semantic_steps:
        operation = (step.get("operation") or "").strip().lower()
        argument = str(step.get("argument") or "")

        if operation == "select":
            object_name, object_id = _semantic_argument_name(argument)
            if object_name:
                for start, end in _find_phrase_spans(tokens, object_name):
                    entities.append(
                        {
                            "label": ENTITY_TYPE_OBJECT,
                            "canonical": object_name,
                            "surface": " ".join(tokens[start:end]),
                            "start": start,
                            "end": end,
                            "match_type": "semantic_select",
                            "source_priority": ENTITY_PRIORITY["semantic_object"],
                            "object_ids": [object_id] if object_id else [],
                        }
                    )
            continue

        if operation in {"relate", "verify rel", "choose rel"}:
            relation_choices, object_name, object_id = _semantic_relation_payload(argument)
            if object_name:
                for start, end in _find_phrase_spans(tokens, object_name):
                    entities.append(
                        {
                            "label": ENTITY_TYPE_OBJECT,
                            "canonical": object_name,
                            "surface": " ".join(tokens[start:end]),
                            "start": start,
                            "end": end,
                            "match_type": f"{operation.replace(' ', '_')}_object",
                            "source_priority": ENTITY_PRIORITY["semantic_object"],
                            "object_ids": [object_id] if object_id else [],
                        }
                    )
            for relation in relation_choices:
                for start, end in _find_phrase_spans(tokens, relation):
                    entities.append(
                        {
                            "label": ENTITY_TYPE_RELATION,
                            "canonical": relation,
                            "surface": " ".join(tokens[start:end]),
                            "start": start,
                            "end": end,
                            "match_type": operation.replace(" ", "_"),
                            "source_priority": ENTITY_PRIORITY["semantic_relation"],
                        }
                    )
            continue

        if operation == "choose name":
            for object_name in _normalize_phrase_choices(argument):
                for start, end in _find_phrase_spans(tokens, object_name):
                    entities.append(
                        {
                            "label": ENTITY_TYPE_OBJECT,
                            "canonical": object_name,
                            "surface": " ".join(tokens[start:end]),
                            "start": start,
                            "end": end,
                            "match_type": "choose_name",
                            "source_priority": ENTITY_PRIORITY["semantic_object"],
                            "object_ids": [],
                        }
                    )
            continue

        for attribute in _attribute_choices_from_step(step):
            for start, end in _find_phrase_spans(tokens, attribute):
                entities.append(
                    {
                        "label": ENTITY_TYPE_ATTRIBUTE,
                        "canonical": attribute,
                        "surface": " ".join(tokens[start:end]),
                        "start": start,
                        "end": end,
                        "match_type": operation.replace(" ", "_") or "semantic_attribute",
                        "source_priority": ENTITY_PRIORITY["semantic_attribute"],
                    }
                )

    return entities


def _dedupe_and_compose_entities(tokens: list[str], candidates: list[dict]) -> list[dict]:
    ordered = sorted(
        candidates,
        key=lambda entity: (
            entity["source_priority"],
            -(entity["end"] - entity["start"]),
            entity["start"],
            TYPE_PRIORITY[entity["label"]],
            entity["canonical"],
        ),
    )

    accepted: list[dict] = []
    occupied = [False] * len(tokens)
    seen: set[tuple[str, int, int, str]] = set()
    for entity in ordered:
        key = (entity["label"], entity["start"], entity["end"], entity["canonical"])
        if key in seen:
            continue
        if any(occupied[idx] for idx in range(entity["start"], entity["end"])):
            continue
        accepted.append(
            {
                key: value
                for key, value in entity.items()
                if key != "source_priority"
            }
        )
        for idx in range(entity["start"], entity["end"]):
            occupied[idx] = True
        seen.add(key)

    accepted.sort(key=lambda entity: (entity["start"], entity["end"], TYPE_PRIORITY[entity["label"]]))
    return accepted


def _build_question_record_from_semantics(question_id: str, item: dict) -> dict:
    tokens = _tokenize_question_for_annotations(item["question"])
    entities = _dedupe_and_compose_entities(tokens, _build_semantic_candidates(tokens, item))
    tags = _bio_tags(tokens, entities)

    return {
        "question_id": str(question_id),
        "image_id": str(item["imageId"]),
        "question": item["question"],
        "normalized_tokens": tokens,
        "bio_tags": tags,
        "entities": entities,
        "answer": item.get("answer"),
    }


def _canonical_label(
    canonical: str,
    object_names: dict[str, set[str]],
    attributes: dict[str, set[str]],
    relations: dict[str, list[dict]],
) -> str | None:
    candidates: list[str] = []
    if canonical in object_names:
        candidates.append(ENTITY_TYPE_OBJECT)
    if canonical in attributes:
        candidates.append(ENTITY_TYPE_ATTRIBUTE)
    if canonical in relations:
        candidates.append(ENTITY_TYPE_RELATION)
    if not candidates:
        return None
    return min(candidates, key=lambda label: TYPE_PRIORITY[label])


def _bio_tags(tokens: list[str], entities: list[dict]) -> list[str]:
    tags = ["O"] * len(tokens)
    for entity in entities:
        label = entity["label"]
        start = entity["start"]
        end = entity["end"]
        tags[start] = f"B-{label}"
        for idx in range(start + 1, end):
            tags[idx] = f"I-{label}"
    return tags


def _resolved_span(tokens: list[str], start: int, end: int) -> tuple[int, int]:
    if start < end and tokens[start] in ARTICLES:
        return start + 1, end
    return start, end


def _build_question_record(question_id: str, item: dict, scene_graph: dict) -> dict:
    object_names, attributes, relations = _build_scene_graph_lexicon(scene_graph)
    vocab = set(object_names.keys()) | set(attributes.keys()) | set(relations.keys())
    linked = link_question_entities(item["question"], vocab)
    tokens = tokenize_for_linking(item["question"])

    entities: list[dict] = []
    for match in linked:
        label = _canonical_label(match.canonical, object_names, attributes, relations)
        if label is None:
            continue
        start, end = _resolved_span(tokens, match.start, match.end)
        if start >= end:
            continue
        payload = {
            "label": label,
            "canonical": match.canonical,
            "surface": match.surface,
            "start": start,
            "end": end,
            "match_type": match.match_type,
        }
        if label == ENTITY_TYPE_OBJECT:
            payload["object_ids"] = sorted(object_names[match.canonical])
        elif label == ENTITY_TYPE_ATTRIBUTE:
            payload["object_ids"] = sorted(attributes[match.canonical])
        else:
            payload["relation_instances"] = relations[match.canonical]
        entities.append(payload)

    entities.sort(key=lambda entity: (entity["start"], -(entity["end"] - entity["start"])))
    tags = _bio_tags(tokens, entities)

    return {
        "question_id": str(question_id),
        "image_id": str(item["imageId"]),
        "question": item["question"],
        "normalized_tokens": tokens,
        "bio_tags": tags,
        "entities": entities,
        "answer": item.get("answer"),
    }


def _summarize_rows(
    split: str,
    output_path: str | Path,
    row_items: list[dict],
    supervision_source: str,
    source_paths: list[str],
) -> dict:
    entity_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    total_entities = 0
    multiword = 0
    no_entity = 0
    with_relations = 0

    for row in row_items:
        entities = row["entities"]
        if not entities:
            no_entity += 1
            continue
        total_entities += len(entities)
        if any((entity["end"] - entity["start"]) > 1 for entity in entities):
            multiword += 1
        if any(entity["label"] == ENTITY_TYPE_RELATION for entity in entities):
            with_relations += 1
        for entity in entities:
            entity_counter[entity["canonical"]] += 1
            label_counter[entity["label"]] += 1

    question_count = max(len(row_items), 1)
    return {
        "split": split,
        "output_path": str(output_path),
        "source_paths": source_paths,
        "question_count": len(row_items),
        "questions_with_entities": len(row_items) - no_entity,
        "questions_without_entities": no_entity,
        "questions_with_entities_pct": round(100.0 * (len(row_items) - no_entity) / question_count, 6),
        "questions_with_multiword_entities_pct": round(100.0 * multiword / question_count, 6),
        "questions_with_relation_entities_pct": round(100.0 * with_relations / question_count, 6),
        "avg_entities_per_question": round(total_entities / question_count, 6),
        "label_distribution": dict(sorted(label_counter.items())),
        "top_canonical_entities": entity_counter.most_common(50),
        "supervision": supervision_source,
        "tagging_scheme": "BIO",
        "entity_types": ENTITY_TYPES,
        "tokenization": (
            "question_annotation_whitespace_tokens"
            if supervision_source == SEMANTIC_SUPERVISION
            else "normalized_whitespace_tokens_from_gqa_preprocessing"
        ),
    }


def _label_schema_payload(supervision_source: str) -> dict:
    return {
        "tagging_scheme": "BIO",
        "entity_types": ENTITY_TYPES,
        "labels": LABELS,
        "label_to_idx": LABEL_TO_IDX,
        "weak_supervision_policy": {
            "source": supervision_source,
            "matching": (
                "official_question_annotations_plus_semantic_program"
                if supervision_source == SEMANTIC_SUPERVISION
                else "deterministic_longest_match"
            ),
            "label_priority": ENTITY_TYPES,
        },
    }


def _prepare_split(
    questions_json: str | Path,
    supervision_source: str,
    scene_graphs: dict[str, dict] | None,
    output_path: str | Path,
    metadata_path: str | Path,
    limit: int | None,
    source_paths: list[str],
) -> None:
    questions = load_gqa_questions(questions_json)
    question_items = list(questions.items())
    if limit is not None:
        question_items = question_items[:limit]

    rows: list[dict] = []
    for question_id, item in question_items:
        if supervision_source == SEMANTIC_SUPERVISION:
            rows.append(_build_question_record_from_semantics(str(question_id), item))
            continue

        image_id = str(item["imageId"])
        if not scene_graphs or image_id not in scene_graphs:
            raise KeyError(
                f"Question {question_id} references imageId={image_id}, but no scene graph was loaded for it."
            )
        rows.append(_build_question_record(str(question_id), item, scene_graphs[image_id]))

    _write_jsonl(output_path, rows)
    metadata = _summarize_rows(
        split=infer_split_name(questions_json),
        output_path=output_path,
        row_items=rows,
        supervision_source=supervision_source,
        source_paths=source_paths,
    )
    _write_json(metadata_path, metadata)


def _prepare_command(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _metadata_dir(processed_dir).mkdir(parents=True, exist_ok=True)

    if args.supervision_source == SCENE_GRAPH_SUPERVISION:
        if not args.scene_graphs:
            raise ValueError("--scene-graphs is required when --supervision-source=scene_graph_alignment")
        all_scene_graphs = load_gqa_scene_graphs(args.scene_graphs)
        source_paths = [str(Path(path)) for path in args.scene_graphs]
    else:
        all_scene_graphs = None
        source_paths = [
            str(Path(args.train_questions)),
            str(Path(args.val_questions)),
        ]

    _write_json(_label_schema_path(processed_dir), _label_schema_payload(args.supervision_source))

    for split, questions_json in (("train", args.train_questions), ("val", args.val_questions)):
        split_scene_graphs = None
        if args.supervision_source == SCENE_GRAPH_SUPERVISION:
            image_ids = {str(item["imageId"]) for item in load_gqa_questions(questions_json).values()}
            split_scene_graphs = {
                image_id: all_scene_graphs[image_id]
                for image_id in image_ids
                if image_id in all_scene_graphs
            }
            missing = image_ids - set(split_scene_graphs.keys())
            if missing:
                raise ValueError(
                    f"Missing {len(missing)} scene graphs for split={split}. "
                    f"First 10: {sorted(missing)[:10]}"
                )

        _prepare_split(
            questions_json=questions_json,
            supervision_source=args.supervision_source,
            scene_graphs=split_scene_graphs,
            output_path=processed_dir / f"{split}.jsonl",
            metadata_path=_split_metadata_path(processed_dir, split),
            limit=args.limit,
            source_paths=source_paths,
        )

    manifest = {
        "dataset_name": "gqa_ner",
        "processed_dir": str(processed_dir),
        "paths": {
            "train": str(processed_dir / "train.jsonl"),
            "val": str(processed_dir / "val.jsonl"),
            "label_schema": str(_label_schema_path(processed_dir)),
        },
        "source_paths": source_paths,
        "supervision": args.supervision_source,
        "entity_types": ENTITY_TYPES,
    }
    _write_json(_package_manifest_path(processed_dir), manifest)

    print(f"Prepared GQA NER dataset under {processed_dir}")
    print("Next step:")
    print(f"  python scripts/validate_gqa_ner_data.py --data-dir {processed_dir}")


def _copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _subset_jsonl(src: str | Path, dst: str | Path, limit: int) -> int:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if written >= limit:
                break
            fout.write(line)
            written += 1
    return written


def _stage_command(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    required = [
        data_dir / "train.jsonl",
        data_dir / "val.jsonl",
        data_dir / "label_schema.json",
        _split_metadata_path(data_dir, "train"),
        _split_metadata_path(data_dir, "val"),
        _package_manifest_path(data_dir),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed GQA NER artifacts:\n" + "\n".join(f"  - {path}" for path in missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _metadata_dir(output_dir).mkdir(exist_ok=True)

    if args.variant == "mini":
        train_count = _subset_jsonl(data_dir / "train.jsonl", output_dir / "train.jsonl", args.mini_n)
        val_count = _subset_jsonl(data_dir / "val.jsonl", output_dir / "val.jsonl", args.mini_n)
    else:
        _copy_file(data_dir / "train.jsonl", output_dir / "train.jsonl")
        _copy_file(data_dir / "val.jsonl", output_dir / "val.jsonl")
        train_count = None
        val_count = None

    _copy_file(data_dir / "label_schema.json", output_dir / "label_schema.json")
    source_manifest = {}
    with open(_package_manifest_path(data_dir)) as f:
        source_manifest = json.load(f)
    for split in ("train", "val"):
        src_meta_path = _split_metadata_path(data_dir, split)
        with open(src_meta_path) as f:
            src_meta = json.load(f)
        rows = _read_jsonl(output_dir / f"{split}.jsonl")
        staged_meta = _summarize_rows(
            split=split,
            output_path=output_dir / f"{split}.jsonl",
            row_items=rows,
            supervision_source=src_meta.get("supervision", source_manifest.get("supervision", SEMANTIC_SUPERVISION)),
            source_paths=src_meta.get("source_paths", source_manifest.get("source_paths", [])),
        )
        staged_meta["source_metadata_path"] = str(src_meta_path)
        _write_json(_split_metadata_path(output_dir, split), staged_meta)

    dataset_id = args.dataset_id or (
        "gqa-ner-data-mini" if args.variant == "mini" else "gqa-ner-data"
    )
    dataset_title = args.dataset_title or (
        f"GQA weakly-supervised NER data (mini {args.mini_n} per split)"
        if args.variant == "mini"
        else "GQA weakly-supervised NER data"
    )
    kaggle_username = _resolve_kaggle_username(args.kaggle_username)
    dataset_owner = kaggle_username or "KAGGLE_USERNAME"

    _write_json(
        output_dir / "dataset-metadata.json",
        {
            "title": dataset_title,
            "id": f"{dataset_owner}/{dataset_id}",
            "licenses": [{"name": "CC BY 4.0"}],
        },
    )

    manifest = {
        "dataset_name": "gqa_ner",
        "variant": args.variant,
        "source_data_dir": str(data_dir),
        "staged_output_dir": str(output_dir),
        "mini_n": args.mini_n if args.variant == "mini" else None,
        "record_counts": {
            "train": train_count,
            "val": val_count,
        },
        "paths": {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "label_schema": "label_schema.json",
        },
        "entity_types": ENTITY_TYPES,
        "supervision": source_manifest.get("supervision", SEMANTIC_SUPERVISION),
        "source_paths": source_manifest.get("source_paths", []),
        "kaggle_owner": kaggle_username,
    }
    _write_json(_package_manifest_path(output_dir), manifest)

    print(f"Staged GQA NER package to {output_dir}")
    print("Next step:")
    print(f"  python scripts/validate_gqa_ner_data.py --data-dir {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a weakly-supervised GQA NER dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build train/val GQA NER JSONL files.")
    prepare.add_argument("--train-questions", required=True)
    prepare.add_argument("--val-questions", required=True)
    prepare.add_argument(
        "--supervision-source",
        choices=[SEMANTIC_SUPERVISION, SCENE_GRAPH_SUPERVISION],
        default=SEMANTIC_SUPERVISION,
    )
    prepare.add_argument("--scene-graphs", nargs="+")
    prepare.add_argument("--processed-dir", required=True)
    prepare.add_argument("--limit", type=int, default=None)
    prepare.set_defaults(handler=_prepare_command)

    stage = subparsers.add_parser("stage", help="Create a Kaggle-ready full or mini GQA NER package.")
    stage.add_argument("--data-dir", required=True)
    stage.add_argument("--output-dir", required=True)
    stage.add_argument("--variant", choices=["full", "mini"], default="full")
    stage.add_argument("--mini-n", type=int, default=500)
    stage.add_argument("--dataset-id")
    stage.add_argument("--dataset-title")
    stage.add_argument("--kaggle-username")
    stage.set_defaults(handler=_stage_command)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv if argv is not None else sys.argv[1:]))
    args.handler(args)


if __name__ == "__main__":
    main()
