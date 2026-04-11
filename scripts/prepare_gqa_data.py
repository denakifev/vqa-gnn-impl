"""
Prepare GQA data artifacts for VQA-GNN (arXiv:2205.11501).

This script handles GQA-specific preprocessing:
  1. Builds an answer vocabulary JSON from GQA balanced training questions.
  2. Converts GQA JSON questions to VQA-compatible format and calls
     scripts/prepare_kg_graphs.py to build per-question KG subgraph HDF5 files.
  3. Optionally validates visual feature HDF5 coverage.

DEVIATION FROM PAPER:
  - Answer vocabulary is built from training set answer frequencies
    (top-N by frequency). The paper's exact vocabulary construction is not
    described; this follows common GQA practice.
  - KG entity extraction uses simple tokenization (not NER/entity linking).
  - Visual features assumed to be 36×2048 bottom-up style (not GQA official).

Required inputs:
  1. GQA balanced questions JSON (train_balanced_questions.json)
  2. ConceptNet Numberbatch embeddings

Usage:
  # Step 1: Build answer vocabulary from training data:
  python scripts/prepare_gqa_data.py \\
    --build-vocab \\
    --questions data/gqa/questions/train_balanced_questions.json \\
    --output-vocab data/gqa/gqa_answer_vocab.json \\
    --top-n 1852

  # Step 2: Build KG graphs for train split:
  python scripts/prepare_gqa_data.py \\
    --build-graphs \\
    --questions data/gqa/questions/train_balanced_questions.json \\
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
    --output data/gqa/knowledge_graphs/train_graphs.h5 \\
    --d-kg 300

  # Step 3: Build KG graphs for val split:
  python scripts/prepare_gqa_data.py \\
    --build-graphs \\
    --questions data/gqa/questions/val_balanced_questions.json \\
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
    --output data/gqa/knowledge_graphs/val_graphs.h5 \\
    --d-kg 300

  # Validate visual features coverage:
  python scripts/prepare_gqa_data.py \\
    --validate-visual \\
    --questions data/gqa/questions/val_balanced_questions.json \\
    --visual-h5 data/gqa/visual_features/val_features.h5

Directory layout after running:
  data/gqa/
  ├── questions/
  │   ├── train_balanced_questions.json
  │   └── val_balanced_questions.json
  ├── visual_features/
  │   ├── train_features.h5   # provided externally
  │   └── val_features.h5
  ├── knowledge_graphs/
  │   ├── train_graphs.h5     # built by this script
  │   └── val_graphs.h5
  └── gqa_answer_vocab.json   # built by this script
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


# ---------------------------------------------------------------------------
# Answer vocabulary builder
# ---------------------------------------------------------------------------


def build_gqa_answer_vocab(questions_path: str, output_path: str, top_n: int = 1852) -> None:
    """
    Build answer vocabulary from GQA training questions.

    GQA has a single ground-truth answer per question. We count answer
    frequencies and keep the top-N most common answers.

    Args:
        questions_path (str): path to train_balanced_questions.json.
        output_path (str): path to write answer_vocab JSON.
        top_n (int): number of answers to keep (default 1852 for GQA balanced).
    """
    questions_path = Path(questions_path)
    if not questions_path.exists():
        print(f"[ERROR] Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading GQA questions from: {questions_path}")
    with open(questions_path) as f:
        q_data = json.load(f)

    counter = Counter()
    for q_id, item in tqdm(q_data.items(), desc="Counting answers"):
        answer = item.get("answer", "")
        if answer:
            counter[answer] += 1

    print(f"Found {len(counter)} unique answers in {len(q_data)} questions.")

    # Keep top-N by frequency
    most_common = counter.most_common(top_n)
    answer_to_idx = {ans: idx for idx, (ans, _) in enumerate(most_common)}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"answer_to_idx": answer_to_idx}, f, indent=2)

    print(f"Saved GQA answer vocabulary: {len(answer_to_idx)} answers → {output_path}")
    if len(most_common) < top_n:
        print(
            f"[WARN] Only {len(most_common)} unique answers found (< {top_n} requested). "
            "Using all unique answers."
        )

    # Coverage stats
    total = len(q_data)
    covered = sum(1 for item in q_data.values() if item.get("answer", "") in answer_to_idx)
    pct = 100.0 * covered / total if total > 0 else 0
    print(f"Answer vocabulary coverage: {covered}/{total} = {pct:.1f}%")


# ---------------------------------------------------------------------------
# Load GQA questions and convert to VQA-compatible format
# ---------------------------------------------------------------------------


def load_gqa_questions(questions_path: str) -> list:
    """
    Load GQA questions and convert to the VQA question-JSON format.

    GQA format: {q_id: {question, imageId, answer, ...}}
    Output format: [{"question_id": q_id, "question": text, "image_id": imageId}, ...]

    Args:
        questions_path (str): path to GQA JSON question file.
    Returns:
        questions (list[dict]): VQA-compatible question list.
    """
    questions_path = Path(questions_path)
    if not questions_path.exists():
        print(f"[ERROR] Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading GQA questions from: {questions_path}")
    with open(questions_path) as f:
        q_data = json.load(f)

    questions = []
    for q_id, item in tqdm(q_data.items(), desc="Converting questions"):
        questions.append(
            {
                "question_id": q_id,
                "question": item["question"],
                "image_id": str(item["imageId"]),
            }
        )
    print(f"Loaded {len(questions)} GQA questions.")
    return questions


def save_vqa_format_questions(questions: list, output_path: str) -> None:
    """Save question list in VQA-compatible JSON format."""
    vqa_fmt = {"questions": questions}
    with open(output_path, "w") as f:
        json.dump(vqa_fmt, f)
    print(f"Saved {len(questions)} questions in VQA format to: {output_path}")


# ---------------------------------------------------------------------------
# Visual feature validation
# ---------------------------------------------------------------------------


def validate_visual_features(questions_path: str, h5_path: str) -> None:
    """Check that each imageId in the questions JSON has a corresponding HDF5 entry."""
    try:
        import h5py
    except ImportError:
        print("[ERROR] h5py is required. Install: pip install h5py", file=sys.stderr)
        sys.exit(1)

    questions_path = Path(questions_path)
    h5_path = Path(h5_path)

    if not h5_path.exists():
        print(f"[ERROR] HDF5 not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading GQA questions from: {questions_path}")
    with open(questions_path) as f:
        q_data = json.load(f)

    image_ids = {str(item["imageId"]) for item in q_data.values()}

    print(f"Validating visual features HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as h5f:
        h5_keys = set(h5f.keys())

    missing = image_ids - h5_keys
    covered = image_ids & h5_keys
    print(f"Unique images in questions: {len(image_ids)}")
    print(f"Images in HDF5:             {len(h5_keys)}")
    print(f"Covered:                    {len(covered)}")
    if missing:
        print(f"[WARN] Missing in HDF5: {len(missing)} images. First 5:")
        for img_id in sorted(missing)[:5]:
            print(f"  {img_id}")
    else:
        print("[OK] All images in questions are present in HDF5.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GQA data artifacts for VQA-GNN."
    )
    parser.add_argument(
        "--build-vocab", action="store_true",
        help="Build GQA answer vocabulary from training questions."
    )
    parser.add_argument(
        "--build-graphs", action="store_true",
        help="Build KG subgraph HDF5 from GQA questions."
    )
    parser.add_argument(
        "--validate-visual", action="store_true",
        help="Validate visual feature HDF5 coverage."
    )

    parser.add_argument("--questions", help="GQA balanced questions JSON.")
    parser.add_argument("--numberbatch", help="ConceptNet Numberbatch embeddings.")
    parser.add_argument("--assertions", default=None, help="ConceptNet assertions CSV.gz.")
    parser.add_argument("--output", help="Output HDF5 path for KG graphs.")
    parser.add_argument(
        "--output-vocab", default="data/gqa/gqa_answer_vocab.json",
        help="Output path for answer vocabulary JSON."
    )
    parser.add_argument(
        "--top-n", type=int, default=1852,
        help="Number of answers to keep in vocabulary (default 1852)."
    )
    parser.add_argument("--d-kg", type=int, default=300)
    parser.add_argument("--num-visual-nodes", type=int, default=36)
    parser.add_argument("--max-kg-nodes", type=int, default=30)
    parser.add_argument("--visual-h5", help="Visual features HDF5 for validation.")

    args = parser.parse_args()

    if not any([args.build_vocab, args.build_graphs, args.validate_visual]):
        parser.error("Specify at least one of: --build-vocab, --build-graphs, --validate-visual")

    if args.build_vocab:
        if not args.questions:
            parser.error("--questions is required with --build-vocab")
        build_gqa_answer_vocab(args.questions, args.output_vocab, args.top_n)

    if args.build_graphs:
        if not args.questions or not args.numberbatch or not args.output:
            parser.error("--questions, --numberbatch, and --output are required with --build-graphs")

        questions = load_gqa_questions(args.questions)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            save_vqa_format_questions(questions, tmp_path)

        script_dir = Path(__file__).parent
        kg_script = script_dir / "prepare_kg_graphs.py"

        cmd = [
            sys.executable, str(kg_script),
            "--questions", tmp_path,
            "--numberbatch", args.numberbatch,
            "--output", args.output,
            "--d-kg", str(args.d_kg),
            "--num-visual-nodes", str(args.num_visual_nodes),
            "--max-kg-nodes", str(args.max_kg_nodes),
        ]
        if args.assertions:
            cmd += ["--assertions", args.assertions]

        print(f"\nCalling prepare_kg_graphs.py for GQA data...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("[ERROR] prepare_kg_graphs.py failed.", file=sys.stderr)
            sys.exit(result.returncode)

        print(f"\n[DONE] GQA KG graphs saved to: {args.output}")

    if args.validate_visual:
        if not args.questions or not args.visual_h5:
            parser.error("--questions and --visual-h5 are required with --validate-visual")
        validate_visual_features(args.questions, args.visual_h5)

    print("\nNext steps:")
    print("  python train.py --config-name baseline_gqa \\")
    print("    datasets=gqa \\")
    print("    datasets.train.data_dir=data/gqa \\")
    print("    datasets.train.answer_vocab_path=data/gqa/gqa_answer_vocab.json \\")
    print("    datasets.val.data_dir=data/gqa \\")
    print("    datasets.val.answer_vocab_path=data/gqa/gqa_answer_vocab.json")


if __name__ == "__main__":
    main()
