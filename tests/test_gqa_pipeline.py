import json
from pathlib import Path

import numpy as np

from src.datasets.gqa_preprocessing import (
    build_answer_vocab,
    build_question_graph,
    build_relation_vocab,
    link_question_entities,
)


def _write_questions(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def _toy_scene_graph() -> dict:
    return {
        "width": 400,
        "height": 300,
        "objects": {
            "1": {
                "name": "fire truck",
                "x": 10,
                "y": 10,
                "w": 50,
                "h": 30,
                "attributes": ["red"],
                "relations": [{"name": "next to", "object": "2"}],
            },
            "2": {
                "name": "street",
                "x": 70,
                "y": 10,
                "w": 120,
                "h": 40,
                "attributes": ["empty"],
                "relations": [{"name": "behind", "object": "3"}],
            },
            "3": {
                "name": "hydrant",
                "x": 210,
                "y": 20,
                "w": 18,
                "h": 28,
                "attributes": ["red"],
                "relations": [],
            },
        },
    }


def test_answer_vocab_is_deterministic_and_contiguous(tmp_path):
    train_path = tmp_path / "train_balanced_questions.json"
    val_path = tmp_path / "val_balanced_questions.json"

    _write_questions(
        train_path,
        {
            "1": {"question": "q1", "imageId": "10", "answer": "Yes"},
            "2": {"question": "q2", "imageId": "11", "answer": " no "},
            "3": {"question": "q3", "imageId": "12", "answer": "Yes"},
        },
    )
    _write_questions(
        val_path,
        {
            "4": {"question": "q4", "imageId": "13", "answer": "maybe"},
            "5": {"question": "q5", "imageId": "14", "answer": "No"},
        },
    )

    answer_to_idx, idx_to_answer, metadata = build_answer_vocab(
        question_paths=[train_path, val_path],
        target_size=3,
    )

    assert idx_to_answer == ["no", "yes", "maybe"]
    assert answer_to_idx == {"no": 0, "yes": 1, "maybe": 2}
    assert metadata["unique_answers_seen"] == 3
    assert metadata["selected_answers"] == 3


def test_entity_linker_prefers_longest_multiword_match():
    vocab = {"fire truck", "truck", "road"}
    matches = link_question_entities("Is the fire truck on the road?", vocab)
    assert [match.canonical for match in matches] == ["fire truck", "road"]
    assert matches[0].match_type == "drop_leading_article"


def test_relation_vocab_contains_forward_and_reverse_edges():
    relation_to_idx, idx_to_relation, metadata = build_relation_vocab({"42": _toy_scene_graph()})

    assert "__no_edge__" in relation_to_idx
    assert "__self__" in relation_to_idx
    assert "__question_visual__" in relation_to_idx
    assert "__question_textual__" in relation_to_idx
    assert "next to" in relation_to_idx
    assert "rev::next to" in relation_to_idx
    assert "behind" in relation_to_idx
    assert "rev::behind" in relation_to_idx
    assert metadata["predicate_count"] == 2
    assert idx_to_relation[relation_to_idx["next to"]] == "next to"


def test_question_graph_uses_scene_graph_relations_and_edge_types():
    scene_graph = _toy_scene_graph()
    relation_to_idx, _, _ = build_relation_vocab({"42": scene_graph})
    glove_embeddings = {
        "fire": np.asarray([1.0, 0.0], dtype=np.float32),
        "truck": np.asarray([0.0, 1.0], dtype=np.float32),
        "street": np.asarray([0.5, 0.5], dtype=np.float32),
        "hydrant": np.asarray([0.25, 0.75], dtype=np.float32),
        "red": np.asarray([0.1, 0.9], dtype=np.float32),
        "empty": np.asarray([0.2, 0.8], dtype=np.float32),
    }
    visual_boxes = np.asarray(
        [
            [10.0, 10.0, 60.0, 40.0],
            [70.0, 10.0, 190.0, 50.0],
            [210.0, 20.0, 228.0, 48.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = build_question_graph(
        question="Is the fire truck next to the street hydrant?",
        scene_graph=scene_graph,
        glove_embeddings=glove_embeddings,
        relation_to_idx=relation_to_idx,
        max_kg_nodes=3,
        num_visual_nodes=4,
        glove_dim=2,
        visual_boxes=visual_boxes,
        actual_visual_nodes=3,
    )

    assert [entity.canonical for entity in result.linked_entities] == [
        "fire truck",
        "street",
        "hydrant",
    ]
    assert result.node_features.shape == (3, 4)
    assert result.actual_visual_nodes == 3
    assert result.aligned_visual_nodes == 3
    assert result.visual_relation_edges == 2
    assert result.textual_relation_edges == 2

    question_idx = 4
    textual_start = 5
    fire_truck_text = textual_start + 0
    street_text = textual_start + 1
    hydrant_text = textual_start + 2

    assert result.adj_matrix[question_idx, 0] == 1.0
    assert result.adj_matrix[question_idx, 1] == 1.0
    assert result.adj_matrix[question_idx, 2] == 1.0
    assert result.adj_matrix[question_idx, 3] == 0.0
    assert (
        result.edge_types[question_idx, 0] == relation_to_idx["__question_visual__"]
    )
    assert (
        result.edge_types[question_idx, fire_truck_text]
        == relation_to_idx["__question_textual__"]
    )
    assert result.edge_types[0, 1] == relation_to_idx["next to"]
    assert result.edge_types[1, 0] == relation_to_idx["rev::next to"]
    assert result.edge_types[1, 2] == relation_to_idx["behind"]
    assert result.edge_types[2, 1] == relation_to_idx["rev::behind"]
    assert result.edge_types[fire_truck_text, street_text] == relation_to_idx["next to"]
    assert result.edge_types[street_text, fire_truck_text] == relation_to_idx["rev::next to"]


def test_question_graph_keeps_scene_graph_nodes_when_question_has_no_match():
    scene_graph = _toy_scene_graph()
    relation_to_idx, _, _ = build_relation_vocab({"42": scene_graph})
    glove_embeddings = {
        "fire": np.asarray([1.0, 0.0], dtype=np.float32),
        "truck": np.asarray([0.0, 1.0], dtype=np.float32),
        "street": np.asarray([0.5, 0.5], dtype=np.float32),
        "hydrant": np.asarray([0.25, 0.75], dtype=np.float32),
    }

    result = build_question_graph(
        question="What time is it?",
        scene_graph=scene_graph,
        glove_embeddings=glove_embeddings,
        relation_to_idx=relation_to_idx,
        max_kg_nodes=2,
        num_visual_nodes=4,
        glove_dim=2,
        visual_boxes=None,
        actual_visual_nodes=0,
    )

    assert result.node_features.shape == (2, 4)
    assert result.selected_object_ids == ["1", "2"]
    assert result.linked_entities == []
    assert result.adj_matrix.shape == (7, 7)
