"""
Validate processed GQA NER artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPECTED_LABELS = {
    "O",
    "B-OBJECT",
    "I-OBJECT",
    "B-ATTRIBUTE",
    "I-ATTRIBUTE",
    "B-RELATION",
    "I-RELATION",
}


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
        return not self.errors


def _load_json(path: Path, report: ValidationReport, description: str) -> dict | None:
    if not path.exists():
        report.fail(f"{description} not found: {path}")
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        report.fail(f"{description} is not valid JSON: {exc}")
        return None
    report.ok(f"{description} loaded: {path}")
    return payload


def validate_label_schema(data_dir: Path, report: ValidationReport) -> None:
    schema = _load_json(data_dir / "label_schema.json", report, "label_schema.json")
    if schema is None:
        return
    labels = schema.get("labels")
    label_to_idx = schema.get("label_to_idx")
    if not isinstance(labels, list) or set(labels) != EXPECTED_LABELS:
        report.fail("label_schema.json has unexpected labels.")
    else:
        report.ok("label schema contains the expected BIO labels.")

    if not isinstance(label_to_idx, dict):
        report.fail("label_schema.json missing label_to_idx.")
    else:
        idx_values = list(label_to_idx.values())
        if sorted(idx_values) != list(range(len(label_to_idx))):
            report.fail("label_to_idx indices are not contiguous.")
        else:
            report.ok("label_to_idx indices are contiguous.")


def _validate_bio_tags(tags: list[str]) -> bool:
    for idx, tag in enumerate(tags):
        if tag not in EXPECTED_LABELS:
            return False
        if tag.startswith("I-"):
            label = tag[2:]
            if idx == 0:
                return False
            prev = tags[idx - 1]
            if prev not in {f"B-{label}", f"I-{label}"}:
                return False
    return True


def validate_split(data_dir: Path, split: str, sample_size: int, report: ValidationReport) -> None:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        report.fail(f"{split}.jsonl not found: {path}")
        return

    rows: list[dict] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                report.fail(f"{split}.jsonl line {line_no} is not valid JSON: {exc}")
                return
            rows.append(row)
    if not rows:
        report.fail(f"{split}.jsonl is empty.")
        return
    report.ok(f"{split}.jsonl contains {len(rows)} records.")

    sample_rows = rows[:sample_size]
    schema_errors = 0
    for row in sample_rows:
        tokens = row.get("normalized_tokens")
        tags = row.get("bio_tags")
        entities = row.get("entities")
        if not isinstance(tokens, list) or not isinstance(tags, list) or not isinstance(entities, list):
            schema_errors += 1
            continue
        if len(tokens) != len(tags):
            schema_errors += 1
            continue
        if not _validate_bio_tags(tags):
            schema_errors += 1
            continue
        for entity in entities:
            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label")
            if not isinstance(start, int) or not isinstance(end, int):
                schema_errors += 1
                break
            if start < 0 or end > len(tokens) or start >= end:
                schema_errors += 1
                break
            if tags[start] != f"B-{label}":
                schema_errors += 1
                break
            for idx in range(start + 1, end):
                if tags[idx] != f"I-{label}":
                    schema_errors += 1
                    break
    if schema_errors:
        report.fail(f"{split}: found {schema_errors} schema/tag consistency errors in sampled rows.")
    else:
        report.ok(f"{split}: sampled rows have consistent token/tag/entity structure.")

    metadata = _load_json(data_dir / "metadata" / f"{split}_metadata.json", report, f"{split} metadata")
    if metadata is None:
        return
    count = metadata.get("question_count")
    if count != len(rows):
        report.fail(f"{split}: metadata question_count={count}, actual rows={len(rows)}.")
    else:
        report.ok(f"{split}: metadata question_count matches JSONL rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed GQA NER artifacts.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report = ValidationReport()

    validate_label_schema(data_dir, report)
    validate_split(data_dir, "train", args.sample_size, report)
    validate_split(data_dir, "val", args.sample_size, report)

    success = report.summary()
    if not success:
        print("\nValidation FAILED.", file=sys.stderr)
        sys.exit(1)

    print("\nValidation passed. GQA NER data looks consistent.")


if __name__ == "__main__":
    main()
