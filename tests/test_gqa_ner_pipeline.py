import json
from pathlib import Path

from scripts.prepare_gqa_ner_data import _build_question_record, _build_question_record_from_semantics


def _toy_scene_graph() -> dict:
    return {
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
                "relations": [],
            },
        }
    }


def test_build_question_record_labels_object_attribute_and_relation():
    row = _build_question_record(
        question_id="q1",
        item={
            "question": "Is the red fire truck next to the street?",
            "imageId": "img1",
            "answer": "yes",
        },
        scene_graph=_toy_scene_graph(),
    )

    assert row["normalized_tokens"] == [
        "is",
        "the",
        "red",
        "fire",
        "truck",
        "next",
        "to",
        "the",
        "street",
    ]
    assert row["bio_tags"] == [
        "O",
        "O",
        "B-ATTRIBUTE",
        "B-OBJECT",
        "I-OBJECT",
        "B-RELATION",
        "I-RELATION",
        "O",
        "B-OBJECT",
    ]
    assert [entity["label"] for entity in row["entities"]] == [
        "ATTRIBUTE",
        "OBJECT",
        "RELATION",
        "OBJECT",
    ]


def test_build_question_record_from_semantics_labels_object_attribute_and_relation():
    row = _build_question_record_from_semantics(
        question_id="q_sem",
        item={
            "question": "Is the red fire truck next to the street?",
            "imageId": "img1",
            "answer": "yes",
            "annotations": {
                "question": {
                    "3:5": "1",
                    "8": "2",
                }
            },
            "semantic": [
                {"operation": "select", "dependencies": [], "argument": "fire truck (1)"},
                {"operation": "verify color", "dependencies": [0], "argument": "red"},
                {"operation": "verify rel", "dependencies": [0], "argument": "street,next to,o (2)"},
            ],
        },
    )

    assert row["normalized_tokens"] == [
        "is",
        "the",
        "red",
        "fire",
        "truck",
        "next",
        "to",
        "the",
        "street",
    ]
    assert row["bio_tags"] == [
        "O",
        "O",
        "B-ATTRIBUTE",
        "B-OBJECT",
        "I-OBJECT",
        "B-RELATION",
        "I-RELATION",
        "O",
        "B-OBJECT",
    ]
    assert [entity["match_type"] for entity in row["entities"]] == [
        "verify_color",
        "question_annotation",
        "verify_rel",
        "question_annotation",
    ]


def test_prepare_script_outputs_jsonl_and_metadata(tmp_path):
    from scripts.prepare_gqa_ner_data import main

    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    train_questions = raw / "train_balanced_questions.json"
    val_questions = raw / "val_balanced_questions.json"
    scene_graphs = raw / "scene_graphs.json"

    with open(train_questions, "w") as f:
        json.dump({"q1": {"question": "Is the red fire truck next to the street?", "imageId": "img1", "answer": "yes"}}, f)
    with open(val_questions, "w") as f:
        json.dump({"q2": {"question": "What color is the fire truck?", "imageId": "img1", "answer": "red"}}, f)
    with open(scene_graphs, "w") as f:
        json.dump({"img1": _toy_scene_graph()}, f)

    output_dir = tmp_path / "processed"
    main(
        [
            "prepare",
            "--train-questions",
            str(train_questions),
            "--val-questions",
            str(val_questions),
            "--scene-graphs",
            str(scene_graphs),
            "--processed-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "label_schema.json").exists()
    assert (output_dir / "metadata" / "train_metadata.json").exists()
    assert (output_dir / "metadata" / "package_manifest.json").exists()


def test_prepare_script_supports_question_semantics_without_scene_graphs(tmp_path):
    from scripts.prepare_gqa_ner_data import main

    raw = tmp_path / "raw_sem"
    raw.mkdir(parents=True, exist_ok=True)
    train_questions = raw / "train_balanced_questions.json"
    val_questions = raw / "val_balanced_questions.json"

    train_payload = {
        "q1": {
            "question": "Is the red fire truck next to the street?",
            "imageId": "img1",
            "answer": "yes",
            "annotations": {"question": {"3:5": "1", "8": "2"}},
            "semantic": [
                {"operation": "select", "dependencies": [], "argument": "fire truck (1)"},
                {"operation": "verify color", "dependencies": [0], "argument": "red"},
                {"operation": "verify rel", "dependencies": [0], "argument": "street,next to,o (2)"},
            ],
        }
    }
    val_payload = {
        "q2": {
            "question": "Is the lid closed or open?",
            "imageId": "img2",
            "answer": "open",
            "annotations": {"question": {"2": "5"}},
            "semantic": [
                {"operation": "select", "dependencies": [], "argument": "lid (5)"},
                {"operation": "choose", "dependencies": [0], "argument": "closed|open"},
            ],
        }
    }

    with open(train_questions, "w") as f:
        json.dump(train_payload, f)
    with open(val_questions, "w") as f:
        json.dump(val_payload, f)

    output_dir = tmp_path / "processed_sem"
    main(
        [
            "prepare",
            "--train-questions",
            str(train_questions),
            "--val-questions",
            str(val_questions),
            "--processed-dir",
            str(output_dir),
        ]
    )

    train_rows = (output_dir / "train.jsonl").read_text().strip().splitlines()
    val_rows = (output_dir / "val.jsonl").read_text().strip().splitlines()
    assert len(train_rows) == 1
    assert len(val_rows) == 1

    with open(output_dir / "label_schema.json") as f:
        schema = json.load(f)
    assert schema["weak_supervision_policy"]["source"] == "question_semantics"

    with open(output_dir / "metadata" / "package_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["supervision"] == "question_semantics"
