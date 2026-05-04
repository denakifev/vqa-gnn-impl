#!/usr/bin/env python3
"""
Question-type stratified analysis for GQA prediction dumps.

This script compares two GQA inference runs (for example baseline vs graph-link)
using per-sample prediction files saved by `inference.py`. It reconstructs
question buckets from the official GQA question annotations and semantic
programs, then reports per-bucket accuracy, delta, and bootstrap confidence
intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


BUCKET_ORDER = [
    "compare",
    "logical",
    "choose",
    "verify",
    "query",
    "relate",
    "other",
]


def _normalize_bucket_label(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    if "compare" in value:
        return "compare"
    if value in {"logical", "logic"}:
        return "logical"
    if "choose" in value:
        return "choose"
    if "verify" in value:
        return "verify"
    if "query" in value:
        return "query"
    if "relate" in value or value == "relation":
        return "relate"
    return None


def infer_question_bucket(item: dict) -> str:
    """
    Infer a coarse question bucket from official GQA annotations.

    Preferred source is `item["types"]` when present. Fallback uses the
    semantic program operations.
    """

    item_types = item.get("types")
    if isinstance(item_types, dict):
        for key in ("structural", "detailed", "semantic", "question"):
            bucket = _normalize_bucket_label(item_types.get(key))
            if bucket is not None:
                return bucket
    elif isinstance(item_types, str):
        bucket = _normalize_bucket_label(item_types)
        if bucket is not None:
            return bucket

    operations = []
    for step in item.get("semantic") or []:
        operation = str(step.get("operation") or "").strip().lower()
        if operation:
            operations.append(operation)

    if not operations:
        return "other"

    if any(op.startswith("compare") for op in operations):
        return "compare"
    if any(
        op in {"and", "or"}
        or op.startswith("exist")
        or op.startswith("same")
        or op.startswith("different")
        for op in operations
    ):
        return "logical"
    if any(op.startswith("choose") for op in operations):
        return "choose"
    if any(op.startswith("verify") for op in operations):
        return "verify"
    if any(op.startswith("query") for op in operations):
        return "query"
    if any(op.startswith("relate") or op.startswith("select") or op.startswith("filter") for op in operations):
        return "relate"
    return "other"


def load_question_buckets(questions_json: Path) -> dict[str, str]:
    with open(questions_json) as f:
        payload = json.load(f)
    return {str(qid): infer_question_bucket(item) for qid, item in payload.items()}


def load_prediction_dir(pred_dir: Path) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    for path in sorted(pred_dir.rglob("output_*.pth")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        sample_id = record.get("sample_id")
        if sample_id is None:
            continue
        outputs[str(sample_id)] = {
            "pred_label": int(record["pred_label"]),
            "label": int(record["label"]),
        }
    return outputs


def bootstrap_delta_ci(
    baseline_correct: np.ndarray,
    candidate_correct: np.ndarray,
    num_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if baseline_correct.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = baseline_correct.size
    deltas = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        indices = rng.integers(0, n, size=n)
        deltas[i] = candidate_correct[indices].mean() - baseline_correct[indices].mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def summarize(
    baseline_preds: dict[str, dict],
    candidate_preds: dict[str, dict],
    buckets: dict[str, str],
    num_bootstrap: int,
    seed: int,
) -> tuple[list[dict], dict]:
    common_ids = sorted(set(baseline_preds) & set(candidate_preds) & set(buckets))
    per_bucket: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for qid in common_ids:
        bucket = buckets[qid]
        baseline_correct = int(baseline_preds[qid]["pred_label"] == baseline_preds[qid]["label"])
        candidate_correct = int(candidate_preds[qid]["pred_label"] == candidate_preds[qid]["label"])
        per_bucket[bucket].append((baseline_correct, candidate_correct))

    rows = []
    total_baseline = []
    total_candidate = []

    for bucket in BUCKET_ORDER:
        pairs = per_bucket.get(bucket, [])
        if not pairs:
            continue
        baseline_arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
        candidate_arr = np.asarray([p[1] for p in pairs], dtype=np.float64)
        ci_low, ci_high = bootstrap_delta_ci(
            baseline_arr,
            candidate_arr,
            num_bootstrap=num_bootstrap,
            seed=seed,
        )
        baseline_acc = float(baseline_arr.mean())
        candidate_acc = float(candidate_arr.mean())
        delta = candidate_acc - baseline_acc
        rows.append(
            {
                "bucket": bucket,
                "n": int(baseline_arr.size),
                "baseline_acc": baseline_acc,
                "candidate_acc": candidate_acc,
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        total_baseline.append(baseline_arr)
        total_candidate.append(candidate_arr)

    baseline_all = np.concatenate(total_baseline) if total_baseline else np.asarray([], dtype=np.float64)
    candidate_all = np.concatenate(total_candidate) if total_candidate else np.asarray([], dtype=np.float64)
    total_ci_low, total_ci_high = bootstrap_delta_ci(
        baseline_all,
        candidate_all,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )
    summary = {
        "num_common_questions": int(len(common_ids)),
        "baseline_acc": float(baseline_all.mean()) if baseline_all.size else 0.0,
        "candidate_acc": float(candidate_all.mean()) if candidate_all.size else 0.0,
        "delta": float(candidate_all.mean() - baseline_all.mean()) if baseline_all.size else 0.0,
        "ci_low": total_ci_low,
        "ci_high": total_ci_high,
    }
    return rows, summary


def write_outputs(output_dir: Path, rows: list[dict], summary: dict, baseline_name: str, candidate_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "question_type_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bucket", "n", "baseline_acc", "candidate_acc", "delta", "ci_low", "ci_high"],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "question_type_metrics.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "baseline_name": baseline_name,
                "candidate_name": candidate_name,
                "summary": summary,
                "rows": rows,
            },
            f,
            indent=2,
        )

    md_path = output_dir / "question_type_report.md"
    lines = [
        "# GQA Question-Type Stratified Analysis",
        "",
        f"- baseline: `{baseline_name}`",
        f"- candidate: `{candidate_name}`",
        f"- common questions: `{summary['num_common_questions']}`",
        f"- baseline accuracy: `{summary['baseline_acc']:.4f}`",
        f"- candidate accuracy: `{summary['candidate_acc']:.4f}`",
        f"- delta: `{summary['delta']:+.4f}`",
        f"- bootstrap 95% CI: `[{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}]`",
        "",
        "| Bucket | N | Baseline | Candidate | Delta | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['bucket']} | {row['n']} | {row['baseline_acc']:.4f} | "
            f"{row['candidate_acc']:.4f} | {row['delta']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question-type stratified analysis for GQA prediction dumps.")
    parser.add_argument("--questions-json", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument("--candidate-name", type=str, default="graph_link")
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    buckets = load_question_buckets(args.questions_json)
    baseline_preds = load_prediction_dir(args.baseline_dir)
    candidate_preds = load_prediction_dir(args.candidate_dir)
    rows, summary = summarize(
        baseline_preds=baseline_preds,
        candidate_preds=candidate_preds,
        buckets=buckets,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
    )
    write_outputs(
        output_dir=args.output_dir,
        rows=rows,
        summary=summary,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )

    print(f"Common questions: {summary['num_common_questions']}")
    print(f"Baseline acc:    {summary['baseline_acc']:.4f}")
    print(f"Candidate acc:   {summary['candidate_acc']:.4f}")
    print(f"Delta:           {summary['delta']:+.4f}")
    print(f"Bootstrap 95% CI [{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}]")
    for row in rows:
        print(
            f"{row['bucket']:>8s}  n={row['n']:5d}  "
            f"base={row['baseline_acc']:.4f}  cand={row['candidate_acc']:.4f}  "
            f"delta={row['delta']:+.4f}"
        )


if __name__ == "__main__":
    main()
