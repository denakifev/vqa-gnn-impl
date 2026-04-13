"""
Utilities for the paper-aligned GQA data pipeline.

For GQA, VQA-GNN does not build a ConceptNet graph. The paper uses a
question-context node together with visual and textual scene graphs:

  - visual SG nodes: official GQA object features
  - textual SG nodes: GloVe embeddings of object name + attributes
  - SG edges: official scene-graph relations

This module is the preprocessing source of truth for that path.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


NODE_TYPE_VISUAL = 0
NODE_TYPE_QUESTION = 1
NODE_TYPE_KG = 2

MAX_ENTITY_NGRAM = 4
DEFAULT_GLOVE_DIM = 300
DEFAULT_IOU_THRESHOLD = 0.5

EDGE_TYPE_NO_EDGE = "__no_edge__"
EDGE_TYPE_SELF = "__self__"
EDGE_TYPE_QUESTION_VISUAL = "__question_visual__"
EDGE_TYPE_QUESTION_TEXTUAL = "__question_textual__"

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

_ARTICLES = frozenset({"a", "an", "the"})
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "done",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "into",
        "onto",
        "over",
        "under",
        "through",
        "across",
        "between",
        "among",
        "about",
        "around",
        "after",
        "before",
        "during",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "as",
        "while",
        "because",
        "that",
        "this",
        "these",
        "those",
        "there",
        "here",
        "it",
        "its",
        "they",
        "them",
        "their",
        "he",
        "she",
        "his",
        "her",
        "we",
        "our",
        "you",
        "your",
        "me",
        "my",
        "i",
        "im",
    }
)
_QUESTION_WORDS = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
    }
)
_GENERIC_TOKENS = frozenset(
    {
        "kind",
        "type",
        "sort",
        "color",
        "number",
        "name",
        "part",
        "side",
        "picture",
        "image",
        "photo",
        "scene",
        "shown",
        "visible",
        "thing",
        "object",
        "item",
    }
)
_RELATION_TOKENS = frozenset(
    {
        "left",
        "right",
        "top",
        "bottom",
        "front",
        "back",
        "behind",
        "under",
        "above",
        "below",
        "inside",
        "outside",
        "near",
        "next",
        "between",
        "beside",
        "around",
    }
)
_SINGLE_TOKEN_BLACKLIST = _STOPWORDS | _QUESTION_WORDS | _GENERIC_TOKENS | _RELATION_TOKENS

_IRREGULAR_SINGULARS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "geese": "goose",
}


@dataclass(frozen=True)
class LinkedEntity:
    canonical: str
    surface: str
    start: int
    end: int
    match_type: str

    @property
    def span_len(self) -> int:
        return self.end - self.start

    @property
    def is_multiword(self) -> bool:
        return " " in self.canonical


@dataclass(frozen=True)
class SceneGraphObject:
    object_id: str
    name: str
    attributes: tuple[str, ...]
    bbox_xyxy: Optional[np.ndarray]
    relations: tuple[tuple[str, str], ...]


@dataclass
class GraphBuildResult:
    node_features: np.ndarray
    adj_matrix: np.ndarray
    edge_types: np.ndarray
    node_types: np.ndarray
    linked_entities: list[LinkedEntity]
    selected_object_ids: list[str]
    actual_visual_nodes: int
    aligned_visual_nodes: int
    visual_relation_edges: int
    textual_relation_edges: int
    truncated_objects: bool


def load_gqa_questions(path: str | Path) -> dict[str, dict]:
    """Load official GQA balanced-split JSON."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected official GQA dict format in {path}, got {type(data).__name__}"
        )
    return {str(question_id): item for question_id, item in data.items()}


def load_gqa_scene_graphs(
    paths: str | Path | Iterable[str | Path],
    image_ids: Optional[set[str]] = None,
) -> dict[str, dict]:
    """
    Load official GQA scene graphs from one or more JSON files.

    Supports either split-specific files (`train_sceneGraphs.json`,
    `val_sceneGraphs.json`) or a single combined JSON.
    """
    if isinstance(paths, (str, Path)):
        path_items = [Path(paths)]
    else:
        path_items = [Path(path) for path in paths]

    target_image_ids = {str(image_id) for image_id in image_ids} if image_ids else None
    scene_graphs: dict[str, dict] = {}
    for path in path_items:
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected official GQA scene-graph dict in {path}, got {type(payload).__name__}"
            )
        for image_id, scene_graph in payload.items():
            image_key = str(image_id)
            if target_image_ids is not None and image_key not in target_image_ids:
                continue
            scene_graphs[image_key] = scene_graph
    return scene_graphs


def infer_split_name(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    if stem.startswith("train_") or "train" in stem:
        return "train"
    if stem.startswith("val_") or "val" in stem:
        return "val"
    if stem.startswith("test_") or "test" in stem:
        return "test"
    return stem


def normalize_answer(answer: str) -> str:
    """
    Light answer normalization for GQA.

    We deliberately avoid the aggressive VQA-v2 answer normalization rules:
    for GQA we want to preserve the official balanced-split answer space and
    only remove case / whitespace noise.
    """
    answer = unicodedata.normalize("NFKC", answer)
    answer = answer.strip().lower()
    answer = _WHITESPACE_RE.sub(" ", answer)
    return answer


def build_answer_vocab(
    question_paths: Iterable[str | Path],
    target_size: int = 1842,
) -> tuple[dict[str, int], list[str], dict]:
    """
    Build a deterministic answer vocabulary from official GQA questions files.

    Answers are sorted by:
      1. descending frequency
      2. lexicographic answer string for deterministic tie-breaking
    """
    counter: Counter[str] = Counter()
    source_files: list[str] = []
    total_answers = 0

    for path in question_paths:
        path = Path(path)
        questions = load_gqa_questions(path)
        source_files.append(str(path))
        for item in questions.values():
            answer = item.get("answer")
            if not answer:
                continue
            counter[normalize_answer(answer)] += 1
            total_answers += 1

    sorted_answers = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    if len(sorted_answers) >= target_size:
        selected = sorted_answers[:target_size]
        truncated = len(sorted_answers) > target_size
    else:
        selected = sorted_answers
        truncated = False

    idx_to_answer = [answer for answer, _ in selected]
    answer_to_idx = {answer: idx for idx, answer in enumerate(idx_to_answer)}
    covered = sum(count for _, count in selected)
    coverage_pct = 100.0 * covered / total_answers if total_answers else 0.0

    metadata = {
        "normalization": "lowercase_strip_collapse_whitespace",
        "source_question_files": source_files,
        "target_size": target_size,
        "unique_answers_seen": len(counter),
        "selected_answers": len(idx_to_answer),
        "selection_policy": (
            "frequency_then_lexicographic"
            if truncated
            else "all_unique_answers_frequency_then_lexicographic"
        ),
        "coverage_pct": round(coverage_pct, 6),
        "truncated": truncated,
        "answer_occurrences": total_answers,
    }

    return answer_to_idx, idx_to_answer, metadata


def save_answer_vocab(
    output_path: str | Path,
    answer_to_idx: dict[str, int],
    idx_to_answer: list[str],
    metadata: dict,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "answer_to_idx": answer_to_idx,
                "idx_to_answer": idx_to_answer,
                "metadata": metadata,
            },
            f,
            indent=2,
        )


def save_relation_vocab(
    output_path: str | Path,
    relation_to_idx: dict[str, int],
    idx_to_relation: list[str],
    metadata: dict,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "relation_to_idx": relation_to_idx,
                "idx_to_relation": idx_to_relation,
                "metadata": metadata,
            },
            f,
            indent=2,
        )


def normalize_linking_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’s", " ")
    text = text.replace("'s", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_for_linking(text: str) -> list[str]:
    normalized = normalize_linking_text(text)
    if not normalized:
        return []
    return normalized.split()


def singularize_token(token: str) -> str:
    if token in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[token]
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _candidate_variants(tokens: list[str]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    variants: list[tuple[str, str]] = []

    def add(tokens_variant: list[str], match_type: str) -> None:
        phrase = " ".join(tokens_variant).strip()
        if not phrase or phrase in seen:
            return
        seen.add(phrase)
        variants.append((phrase, match_type))

    add(tokens, "exact")

    if len(tokens) > 1 and tokens[0] in _ARTICLES:
        add(tokens[1:], "drop_leading_article")

    singular_tokens = list(tokens)
    singular_tokens[-1] = singularize_token(singular_tokens[-1])
    if singular_tokens[-1] != tokens[-1]:
        add(singular_tokens, "singularize_last_token")

    if len(tokens) > 1 and tokens[0] in _ARTICLES:
        singular_article_tokens = list(tokens[1:])
        singular_article_tokens[-1] = singularize_token(singular_article_tokens[-1])
        add(singular_article_tokens, "drop_article_and_singularize_last_token")

    return variants


def _span_is_candidate(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if all(token in _STOPWORDS or token in _QUESTION_WORDS for token in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) <= 1 or token.isdigit():
            return False
        if token in _SINGLE_TOKEN_BLACKLIST:
            return False
    return True


def link_question_entities(
    question: str,
    vocab_set: set[str],
    max_ngram: int = MAX_ENTITY_NGRAM,
) -> list[LinkedEntity]:
    """
    Deterministically link question text to scene-graph object names / attributes.

    Strategy:
      1. Normalize question text.
      2. Generate n-gram candidates up to `max_ngram`.
      3. Prefer longest matches and non-overlapping spans.
      4. Fall back to singularized tail-token variants where useful.
    """
    tokens = tokenize_for_linking(question)
    if not tokens:
        return []

    spans: list[tuple[int, int, list[str]]] = []
    n_tokens = len(tokens)
    for start in range(n_tokens):
        max_end = min(n_tokens, start + max_ngram)
        for end in range(max_end, start, -1):
            span_tokens = tokens[start:end]
            if _span_is_candidate(span_tokens):
                spans.append((start, end, span_tokens))

    spans.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[1]))

    occupied_positions: set[int] = set()
    seen_entities: set[str] = set()
    matches: list[LinkedEntity] = []

    for start, end, span_tokens in spans:
        if any(position in occupied_positions for position in range(start, end)):
            continue

        matched_entity = None
        matched_type = None
        for candidate, match_type in _candidate_variants(span_tokens):
            if candidate in vocab_set:
                matched_entity = candidate
                matched_type = match_type
                break

        if matched_entity is None or matched_entity in seen_entities:
            continue

        matches.append(
            LinkedEntity(
                canonical=matched_entity,
                surface=" ".join(span_tokens),
                start=start,
                end=end,
                match_type=matched_type or "exact",
            )
        )
        seen_entities.add(matched_entity)
        occupied_positions.update(range(start, end))

    matches.sort(key=lambda item: (item.start, -item.span_len, item.canonical))
    return matches


def _scene_graph_object_sort_key(object_id: str) -> tuple[int, int | str]:
    if object_id.isdigit():
        return (0, int(object_id))
    return (1, object_id)


def _coerce_bbox_xyxy(item: dict) -> Optional[np.ndarray]:
    try:
        x = float(item["x"])
        y = float(item["y"])
        w = float(item["w"])
        h = float(item["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def _normalize_phrase_list(values) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        phrase = normalize_linking_text(value)
        if phrase:
            normalized.append(phrase)
    return tuple(dict.fromkeys(normalized))


def parse_scene_graph_objects(scene_graph: dict) -> list[SceneGraphObject]:
    objects_payload = scene_graph.get("objects", {})
    if not isinstance(objects_payload, dict):
        return []

    objects: list[SceneGraphObject] = []
    for object_id in sorted((str(key) for key in objects_payload.keys()), key=_scene_graph_object_sort_key):
        payload = objects_payload.get(object_id, {})
        if not isinstance(payload, dict):
            continue
        name_raw = payload.get("name", "")
        if not isinstance(name_raw, str):
            name_raw = str(name_raw)
        name = normalize_linking_text(name_raw)
        if not name:
            continue

        attributes = _normalize_phrase_list(payload.get("attributes", []))
        bbox_xyxy = _coerce_bbox_xyxy(payload)

        relations_payload = payload.get("relations", [])
        relations: list[tuple[str, str]] = []
        if isinstance(relations_payload, list):
            for relation in relations_payload:
                if not isinstance(relation, dict):
                    continue
                predicate_raw = relation.get("name", "")
                target_raw = relation.get("object")
                if not isinstance(predicate_raw, str):
                    continue
                predicate = normalize_linking_text(predicate_raw)
                if not predicate or target_raw is None:
                    continue
                relations.append((predicate, str(target_raw)))

        objects.append(
            SceneGraphObject(
                object_id=str(object_id),
                name=name,
                attributes=attributes,
                bbox_xyxy=bbox_xyxy,
                relations=tuple(relations),
            )
        )
    return objects


def _build_phrase_to_objects(
    objects: list[SceneGraphObject],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    name_to_objects: dict[str, set[str]] = defaultdict(set)
    attr_to_objects: dict[str, set[str]] = defaultdict(set)
    for obj in objects:
        name_to_objects[obj.name].add(obj.object_id)
        for attr in obj.attributes:
            attr_to_objects[attr].add(obj.object_id)
    return name_to_objects, attr_to_objects


def match_question_to_scene_graph_objects(
    question: str,
    objects: list[SceneGraphObject],
) -> tuple[list[LinkedEntity], dict[str, int]]:
    name_to_objects, attr_to_objects = _build_phrase_to_objects(objects)
    vocab = set(name_to_objects.keys()) | set(attr_to_objects.keys())
    linked_entities = link_question_entities(question, vocab)

    scores: Counter[str] = Counter()
    for entity in linked_entities:
        for object_id in name_to_objects.get(entity.canonical, ()):
            scores[object_id] += 100 + 10 * entity.span_len
        for object_id in attr_to_objects.get(entity.canonical, ()):
            scores[object_id] += 25 + 5 * entity.span_len

    return linked_entities, dict(scores)


def select_scene_graph_objects(
    question: str,
    scene_graph: dict,
    max_objects: int,
) -> tuple[list[SceneGraphObject], list[LinkedEntity], bool]:
    objects = parse_scene_graph_objects(scene_graph)
    if max_objects <= 0 or not objects:
        return [], [], False

    objects_by_id = {obj.object_id: obj for obj in objects}
    linked_entities, scores = match_question_to_scene_graph_objects(question, objects)

    incoming_neighbors: dict[str, set[str]] = defaultdict(set)
    degrees: Counter[str] = Counter()
    for obj in objects:
        for _, target_id in obj.relations:
            degrees[obj.object_id] += 1
            incoming_neighbors[target_id].add(obj.object_id)
            degrees[target_id] += 1

    selected_ids: list[str] = []
    seen_ids: set[str] = set()

    matched_ids = sorted(
        scores.keys(),
        key=lambda object_id: (
            -scores[object_id],
            -degrees[object_id],
            _scene_graph_object_sort_key(object_id),
        ),
    )
    for object_id in matched_ids:
        if object_id in seen_ids:
            continue
        selected_ids.append(object_id)
        seen_ids.add(object_id)
        if len(selected_ids) >= max_objects:
            break

    if len(selected_ids) < max_objects and matched_ids:
        neighbor_scores: Counter[str] = Counter()
        for object_id in matched_ids:
            obj = objects_by_id[object_id]
            for _, target_id in obj.relations:
                if target_id in objects_by_id:
                    neighbor_scores[target_id] += 1
            for source_id in incoming_neighbors.get(object_id, ()):
                if source_id in objects_by_id:
                    neighbor_scores[source_id] += 1

        neighbor_ids = sorted(
            neighbor_scores.keys(),
            key=lambda object_id: (
                -neighbor_scores[object_id],
                -scores.get(object_id, 0),
                -degrees[object_id],
                _scene_graph_object_sort_key(object_id),
            ),
        )
        for object_id in neighbor_ids:
            if object_id in seen_ids:
                continue
            selected_ids.append(object_id)
            seen_ids.add(object_id)
            if len(selected_ids) >= max_objects:
                break

    if len(selected_ids) < max_objects:
        for obj in objects:
            if obj.object_id in seen_ids:
                continue
            selected_ids.append(obj.object_id)
            seen_ids.add(obj.object_id)
            if len(selected_ids) >= max_objects:
                break

    selected_objects = [objects_by_id[object_id] for object_id in selected_ids]
    truncated = len(objects) > len(selected_objects)
    return selected_objects, linked_entities, truncated


def collect_glove_token_vocab_from_scene_graphs(
    scene_graphs: dict[str, dict],
) -> set[str]:
    token_vocab: set[str] = set()
    for scene_graph in scene_graphs.values():
        for obj in parse_scene_graph_objects(scene_graph):
            token_vocab.update(tokenize_for_linking(obj.name))
            for attr in obj.attributes:
                token_vocab.update(tokenize_for_linking(attr))
    return token_vocab


def load_glove_subset(
    path: str | Path,
    required_tokens: set[str],
    target_dim: int = DEFAULT_GLOVE_DIM,
) -> dict[str, np.ndarray]:
    """
    Load only the GloVe rows needed for the selected GQA scene-graph tokens.
    """
    required_tokens = {token for token in required_tokens if token}
    if not required_tokens:
        return {}

    glove_path = Path(path)
    opener = gzip.open if str(glove_path).endswith(".gz") else open
    embeddings: dict[str, np.ndarray] = {}
    with opener(glove_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe subset", unit="lines"):
            parts = line.rstrip().split(" ")
            if len(parts) <= target_dim:
                continue
            token_raw = parts[0]
            token = normalize_linking_text(token_raw)
            if not token or " " in token or token not in required_tokens:
                continue
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if vector.shape[0] < target_dim:
                continue
            embeddings[token] = vector[:target_dim]
    return embeddings


def _average_token_embeddings(
    tokens: list[str],
    glove_embeddings: dict[str, np.ndarray],
    glove_dim: int,
) -> np.ndarray:
    vectors = [glove_embeddings[token] for token in tokens if token in glove_embeddings]
    if not vectors:
        return np.zeros(glove_dim, dtype=np.float32)
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


def build_scene_graph_node_feature(
    obj: SceneGraphObject,
    glove_embeddings: dict[str, np.ndarray],
    glove_dim: int,
) -> np.ndarray:
    """
    Paper-like textual SG node feature:
      concat(GloVe(object_name), GloVe(mean_attributes))
    """
    name_vec = _average_token_embeddings(tokenize_for_linking(obj.name), glove_embeddings, glove_dim)
    attr_tokens: list[str] = []
    for attr in obj.attributes:
        attr_tokens.extend(tokenize_for_linking(attr))
    attr_vec = _average_token_embeddings(attr_tokens, glove_embeddings, glove_dim)
    return np.concatenate([name_vec, attr_vec], axis=0).astype(np.float32)


def build_relation_vocab(
    scene_graphs: dict[str, dict],
) -> tuple[dict[str, int], list[str], dict]:
    predicates: set[str] = set()
    image_count = 0
    relation_count = 0
    for scene_graph in scene_graphs.values():
        image_count += 1
        for obj in parse_scene_graph_objects(scene_graph):
            for predicate, _ in obj.relations:
                predicates.add(predicate)
                relation_count += 1

    idx_to_relation = [
        EDGE_TYPE_NO_EDGE,
        EDGE_TYPE_SELF,
        EDGE_TYPE_QUESTION_VISUAL,
        EDGE_TYPE_QUESTION_TEXTUAL,
    ]
    for predicate in sorted(predicates):
        idx_to_relation.append(predicate)
        idx_to_relation.append(f"rev::{predicate}")

    relation_to_idx = {relation: idx for idx, relation in enumerate(idx_to_relation)}
    metadata = {
        "selection_policy": "all_unique_scene_graph_predicates_lexicographic",
        "special_relations": [
            EDGE_TYPE_NO_EDGE,
            EDGE_TYPE_SELF,
            EDGE_TYPE_QUESTION_VISUAL,
            EDGE_TYPE_QUESTION_TEXTUAL,
        ],
        "predicate_count": len(predicates),
        "relation_count_total": len(idx_to_relation),
        "scene_graph_image_count": image_count,
        "scene_graph_relation_instances": relation_count,
    }
    return relation_to_idx, idx_to_relation, metadata


class GQAObjectFeatureStore:
    """
    Accessor for official GQA object features and bounding boxes.

    Expected files:
      - gqa_objects.h5 or directory of gqa_objects_*.h5
      - gqa_objects_info.json
    """

    def __init__(self, features_path: str | Path, info_json: str | Path):
        if h5py is None:
            raise ImportError("h5py is required for GQAObjectFeatureStore.")

        self.features_path = Path(features_path)
        self.info_json = Path(info_json)
        with open(self.info_json) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected dict in {self.info_json}, got {type(payload).__name__}."
            )
        self._info = {str(image_id): value for image_id, value in payload.items()}

        if self.features_path.is_dir():
            self._shard_paths = {
                int(path.stem.rsplit("_", 1)[-1]): path
                for path in sorted(self.features_path.glob("gqa_objects_*.h5"))
            }
            if not self._shard_paths:
                raise FileNotFoundError(
                    f"No shard files matching gqa_objects_*.h5 found in {self.features_path}"
                )
        elif self.features_path.is_file():
            self._shard_paths = {None: self.features_path}
        else:
            raise FileNotFoundError(f"GQA object feature input not found: {self.features_path}")

        self._open_handles: dict[int | None, h5py.File] = {}

    def close(self) -> None:
        for handle in self._open_handles.values():
            handle.close()
        self._open_handles.clear()

    def __enter__(self) -> "GQAObjectFeatureStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _resolve_shard_key(self, file_id) -> int | None:
        if None in self._shard_paths:
            return None
        if file_id is None:
            raise KeyError("Object feature metadata is missing 'file' for sharded GQA input.")
        return int(file_id)

    def _open_shard(self, shard_key: int | None):
        if shard_key not in self._shard_paths:
            raise KeyError(f"GQA feature shard file id={shard_key} not found.")
        if shard_key not in self._open_handles:
            self._open_handles[shard_key] = h5py.File(self._shard_paths[shard_key], "r")
        return self._open_handles[shard_key]

    @staticmethod
    def _get_dataset(handle, *names):
        for name in names:
            if name in handle:
                return handle[name]
        raise KeyError(f"Expected one of datasets {names}, found keys: {list(handle.keys())[:10]}")

    def get_boxes(self, image_id: str) -> tuple[np.ndarray, int]:
        image_id = str(image_id)
        meta = self._info.get(image_id)
        if meta is None:
            raise KeyError(f"imageId={image_id} missing from {self.info_json}")
        file_id = meta.get("file")
        idx = meta.get("idx")
        if idx is None:
            raise KeyError(f"imageId={image_id} has no idx entry in {self.info_json}")

        shard_key = self._resolve_shard_key(file_id)
        handle = self._open_shard(shard_key)
        boxes_ds = self._get_dataset(handle, "bboxes", "boxes")
        boxes = np.asarray(boxes_ds[idx], dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            raise ValueError(f"imageId={image_id} has unexpected box shape {boxes.shape}")

        objects_num = meta.get("objectsNum")
        if isinstance(objects_num, int) and objects_num > 0:
            boxes = boxes[:objects_num]

        return boxes[:, :4], min(int(meta.get("objectsNum", boxes.shape[0])), boxes.shape[0])


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def align_scene_graph_objects_to_visual_boxes(
    objects: list[SceneGraphObject],
    feature_boxes: Optional[np.ndarray],
    num_visual_nodes: int,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, int]:
    if feature_boxes is None or feature_boxes.size == 0:
        return {}

    usable_boxes = np.asarray(feature_boxes[:num_visual_nodes], dtype=np.float32)
    candidates: list[tuple[float, str, int]] = []
    for obj in objects:
        if obj.bbox_xyxy is None:
            continue
        for visual_idx, box in enumerate(usable_boxes):
            iou = _bbox_iou(obj.bbox_xyxy, box)
            if iou >= iou_threshold:
                candidates.append((iou, obj.object_id, visual_idx))

    candidates.sort(key=lambda item: (-item[0], _scene_graph_object_sort_key(item[1]), item[2]))

    alignment: dict[str, int] = {}
    used_visual_indices: set[int] = set()
    for _, object_id, visual_idx in candidates:
        if object_id in alignment or visual_idx in used_visual_indices:
            continue
        alignment[object_id] = visual_idx
        used_visual_indices.add(visual_idx)
    return alignment


def build_question_graph(
    question: str,
    scene_graph: dict,
    glove_embeddings: dict[str, np.ndarray],
    relation_to_idx: dict[str, int],
    max_kg_nodes: int,
    num_visual_nodes: int,
    glove_dim: int,
    visual_boxes: Optional[np.ndarray] = None,
    actual_visual_nodes: Optional[int] = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> GraphBuildResult:
    """
    Build a GQA graph in the paper-like scene-graph regime.

    Graph layout:
      visual nodes (official object features) | question node | textual SG nodes
    """
    all_objects = parse_scene_graph_objects(scene_graph)
    selected_objects, linked_entities, truncated_objects = select_scene_graph_objects(
        question=question,
        scene_graph=scene_graph,
        max_objects=max_kg_nodes,
    )
    selected_object_ids = [obj.object_id for obj in selected_objects]
    selected_object_id_set = set(selected_object_ids)
    n_textual_nodes = len(selected_objects)

    d_kg = glove_dim * 2
    if n_textual_nodes:
        node_features = np.stack(
            [
                build_scene_graph_node_feature(obj, glove_embeddings, glove_dim)
                for obj in selected_objects
            ],
            axis=0,
        ).astype(np.float32)
    else:
        node_features = np.zeros((0, d_kg), dtype=np.float32)

    n_total = num_visual_nodes + 1 + n_textual_nodes
    question_idx = num_visual_nodes
    textual_start = num_visual_nodes + 1

    adj = np.zeros((n_total, n_total), dtype=np.float32)
    edge_types = np.zeros((n_total, n_total), dtype=np.int32)
    np.fill_diagonal(adj, 1.0)
    np.fill_diagonal(edge_types, relation_to_idx[EDGE_TYPE_SELF])

    if visual_boxes is not None:
        resolved_actual_visual_nodes = min(
            num_visual_nodes,
            actual_visual_nodes if actual_visual_nodes is not None else visual_boxes.shape[0],
        )
    else:
        resolved_actual_visual_nodes = min(
            num_visual_nodes,
            actual_visual_nodes if actual_visual_nodes is not None else num_visual_nodes,
        )

    for visual_idx in range(resolved_actual_visual_nodes):
        adj[question_idx, visual_idx] = 1.0
        adj[visual_idx, question_idx] = 1.0
        edge_types[question_idx, visual_idx] = relation_to_idx[EDGE_TYPE_QUESTION_VISUAL]
        edge_types[visual_idx, question_idx] = relation_to_idx[EDGE_TYPE_QUESTION_VISUAL]

    for text_offset in range(n_textual_nodes):
        text_idx = textual_start + text_offset
        adj[question_idx, text_idx] = 1.0
        adj[text_idx, question_idx] = 1.0
        edge_types[question_idx, text_idx] = relation_to_idx[EDGE_TYPE_QUESTION_TEXTUAL]
        edge_types[text_idx, question_idx] = relation_to_idx[EDGE_TYPE_QUESTION_TEXTUAL]

    full_alignment = align_scene_graph_objects_to_visual_boxes(
        objects=all_objects,
        feature_boxes=visual_boxes,
        num_visual_nodes=num_visual_nodes,
        iou_threshold=iou_threshold,
    )

    visual_relation_edges = 0
    seen_visual_edges: set[tuple[int, int, str]] = set()
    for obj in all_objects:
        src_visual = full_alignment.get(obj.object_id)
        if src_visual is None:
            continue
        for predicate, target_id in obj.relations:
            dst_visual = full_alignment.get(target_id)
            if dst_visual is None or src_visual == dst_visual or predicate not in relation_to_idx:
                continue
            edge_key = (src_visual, dst_visual, predicate)
            if edge_key in seen_visual_edges:
                continue
            seen_visual_edges.add(edge_key)
            adj[src_visual, dst_visual] = 1.0
            adj[dst_visual, src_visual] = 1.0
            edge_types[src_visual, dst_visual] = relation_to_idx[predicate]
            edge_types[dst_visual, src_visual] = relation_to_idx[f"rev::{predicate}"]
            visual_relation_edges += 1

    textual_index = {
        object_id: textual_start + idx for idx, object_id in enumerate(selected_object_ids)
    }
    textual_relation_edges = 0
    seen_textual_edges: set[tuple[int, int, str]] = set()
    for obj in selected_objects:
        src_text = textual_index[obj.object_id]
        for predicate, target_id in obj.relations:
            if target_id not in selected_object_id_set or predicate not in relation_to_idx:
                continue
            dst_text = textual_index[target_id]
            if src_text == dst_text:
                continue
            edge_key = (src_text, dst_text, predicate)
            if edge_key in seen_textual_edges:
                continue
            seen_textual_edges.add(edge_key)
            adj[src_text, dst_text] = 1.0
            adj[dst_text, src_text] = 1.0
            edge_types[src_text, dst_text] = relation_to_idx[predicate]
            edge_types[dst_text, src_text] = relation_to_idx[f"rev::{predicate}"]
            textual_relation_edges += 1

    node_types = np.asarray(
        [NODE_TYPE_VISUAL] * num_visual_nodes
        + [NODE_TYPE_QUESTION]
        + [NODE_TYPE_KG] * n_textual_nodes,
        dtype=np.int32,
    )

    return GraphBuildResult(
        node_features=node_features,
        adj_matrix=adj,
        edge_types=edge_types,
        node_types=node_types,
        linked_entities=linked_entities,
        selected_object_ids=selected_object_ids,
        actual_visual_nodes=resolved_actual_visual_nodes,
        aligned_visual_nodes=len(set(full_alignment.values())),
        visual_relation_edges=visual_relation_edges,
        textual_relation_edges=textual_relation_edges,
        truncated_objects=truncated_objects,
    )
