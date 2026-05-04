import json
from pathlib import Path

import torch

from scripts.analyze_gqa_question_types import (
    infer_question_bucket,
    load_prediction_dir,
    summarize,
)


def test_infer_question_bucket_prefers_types_when_available():
    item = {
        "types": {"structural": "compare"},
        "semantic": [{"operation": "verify rel"}],
    }
    assert infer_question_bucket(item) == "compare"


def test_infer_question_bucket_falls_back_to_semantic_program():
    item = {
        "semantic": [
            {"operation": "select"},
            {"operation": "verify rel"},
        ]
    }
    assert infer_question_bucket(item) == "verify"


def test_summarize_computes_bucket_delta(tmp_path: Path):
    questions = {
        "q1": {"semantic": [{"operation": "verify color"}]},
        "q2": {"semantic": [{"operation": "compare"}]},
    }
    with open(tmp_path / "questions.json", "w") as f:
        json.dump(questions, f)

    baseline_dir = tmp_path / "baseline" / "val"
    candidate_dir = tmp_path / "candidate" / "val"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    torch.save({"sample_id": "q1", "pred_label": 0, "label": 1}, baseline_dir / "output_0.pth")
    torch.save({"sample_id": "q2", "pred_label": 1, "label": 1}, baseline_dir / "output_1.pth")
    torch.save({"sample_id": "q1", "pred_label": 1, "label": 1}, candidate_dir / "output_0.pth")
    torch.save({"sample_id": "q2", "pred_label": 1, "label": 1}, candidate_dir / "output_1.pth")

    baseline_preds = load_prediction_dir(tmp_path / "baseline")
    candidate_preds = load_prediction_dir(tmp_path / "candidate")
    buckets = {qid: infer_question_bucket(item) for qid, item in questions.items()}

    rows, summary = summarize(
        baseline_preds=baseline_preds,
        candidate_preds=candidate_preds,
        buckets=buckets,
        num_bootstrap=100,
        seed=42,
    )

    by_bucket = {row["bucket"]: row for row in rows}
    assert by_bucket["verify"]["delta"] > 0
    assert by_bucket["compare"]["delta"] == 0
    assert summary["delta"] > 0
