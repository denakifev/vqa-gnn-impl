"""
Prepare VCR data artifacts for VQA-GNN (arXiv:2205.11501).

This script handles VCR-specific preprocessing:
  1. Extracts question texts from VCR JSONL annotations and converts them to
     the question-JSON format expected by scripts/prepare_kg_graphs.py.
  2. Calls prepare_kg_graphs.py to build per-question KG subgraph HDF5 files
     (keyed by annot_id).
  3. Optionally validates visual feature HDF5 coverage against JSONL annotations.

DEVIATION FROM PAPER:
  - KG is built from question text only (same graph for all 4 candidates).
    The paper may build candidate-aware graphs; this is not described.
  - VCR object references ([0, 2] → "person car") are resolved to category names
    for entity extraction. The paper likely uses region-level entity grounding.
  - Entity extraction uses simple tokenization (stopword filter), not NER/linking.

Required inputs:
  1. VCR JSONL annotation file (train.jsonl or val.jsonl)
  2. ConceptNet Numberbatch embeddings (same file as used for VQA v2)
  Optional:
  3. ConceptNet assertions CSV (for richer KG edge structure)

Usage:
  # Step 1: Extract VCR questions and build KG graphs (train split):
  python scripts/prepare_vcr_data.py \\
    --jsonl data/vcr/train.jsonl \\
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
    --output data/vcr/knowledge_graphs/train_graphs.h5 \\
    --d-kg 300

  # With ConceptNet assertions (better edge structure):
  python scripts/prepare_vcr_data.py \\
    --jsonl data/vcr/val.jsonl \\
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
    --assertions data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \\
    --output data/vcr/knowledge_graphs/val_graphs.h5 \\
    --d-kg 300

  # Validate visual features coverage:
  python scripts/prepare_vcr_data.py \\
    --jsonl data/vcr/val.jsonl \\
    --validate-visual data/vcr/visual_features/val_features.h5

Directory layout after running:
  data/vcr/
  ├── train.jsonl
  ├── val.jsonl
  ├── visual_features/
  │   ├── train_features.h5   # provided externally (R2C Faster RCNN features)
  │   └── val_features.h5
  └── knowledge_graphs/
      ├── train_graphs.h5     # built by this script
      └── val_graphs.h5
"""

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


# ---------------------------------------------------------------------------
# VCR object reference resolution
# ---------------------------------------------------------------------------


def vcr_tokens_to_text(tokens: list, objects: list) -> str:
    """
    Convert a VCR token list to plain text, resolving object references.

    Args:
        tokens (list): VCR token list (strings or int lists).
        objects (list[str]): list of object category names for the image.
    Returns:
        text (str): plain text string.
    """
    parts = []
    for tok in tokens:
        if isinstance(tok, list):
            names = [objects[i] for i in tok if i < len(objects)]
            parts.append(" and ".join(names) if names else "object")
        else:
            parts.append(str(tok))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Convert VCR JSONL to question-JSON format for prepare_kg_graphs.py
# ---------------------------------------------------------------------------


def load_vcr_questions(jsonl_path: str) -> list:
    """
    Load VCR annotations and convert to a flat list of (annot_id, question_text).

    Args:
        jsonl_path (str): path to VCR JSONL file.
    Returns:
        questions (list[dict]): each dict has 'question_id' and 'question'.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"[ERROR] JSONL file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    questions = []
    print(f"Loading VCR annotations from: {jsonl_path}")
    with open(jsonl_path) as f:
        for line in tqdm(f, desc="Parsing JSONL"):
            item = json.loads(line)
            objects = item.get("objects", [])
            q_text = vcr_tokens_to_text(item["question"], objects)
            questions.append(
                {
                    "question_id": item["annot_id"],  # use annot_id as question_id key
                    "question": q_text,
                    "image_id": item["img_fn"],       # not used for KG building
                }
            )
    print(f"Loaded {len(questions)} VCR questions from {jsonl_path.name}.")
    return questions


def save_vqa_format_questions(questions: list, output_path: str) -> None:
    """
    Save question list in VQA-compatible JSON format for prepare_kg_graphs.py.

    prepare_kg_graphs.py expects:
        { "questions": [{"question_id": ..., "question": ..., "image_id": ...}, ...] }

    Args:
        questions (list[dict]): output of load_vcr_questions.
        output_path (str): path to write the temp JSON.
    """
    vqa_fmt = {"questions": questions}
    with open(output_path, "w") as f:
        json.dump(vqa_fmt, f)
    print(f"Saved {len(questions)} questions in VQA format to: {output_path}")


# ---------------------------------------------------------------------------
# Visual feature validation
# ---------------------------------------------------------------------------


def validate_visual_features(jsonl_path: str, h5_path: str) -> None:
    """
    Check that each img_fn in the JSONL has a corresponding entry in the HDF5.

    Args:
        jsonl_path (str): path to VCR JSONL file.
        h5_path (str): path to visual features HDF5.
    """
    try:
        import h5py
    except ImportError:
        print("[ERROR] h5py is required. Install: pip install h5py", file=sys.stderr)
        sys.exit(1)

    print(f"Validating visual features: {h5_path}")
    jsonl_path = Path(jsonl_path)
    h5_path = Path(h5_path)

    if not h5_path.exists():
        print(f"[ERROR] HDF5 not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    img_fns = set()
    with open(jsonl_path) as f:
        for line in tqdm(f, desc="Reading img_fns"):
            item = json.loads(line)
            img_fns.add(item["img_fn"])

    with h5py.File(h5_path, "r") as h5f:
        h5_keys = set(h5f.keys())

    missing = img_fns - h5_keys
    covered = img_fns & h5_keys
    print(f"Images in JSONL: {len(img_fns)}")
    print(f"Images in HDF5:  {len(h5_keys)}")
    print(f"Covered:         {len(covered)}")
    if missing:
        print(f"[WARN] Missing in HDF5: {len(missing)} images. First 5:")
        for fn in sorted(missing)[:5]:
            print(f"  {fn}")
    else:
        print("[OK] All images in JSONL are present in HDF5.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VCR KG graphs for VQA-GNN."
    )
    parser.add_argument(
        "--jsonl", required=False,
        help="VCR annotation JSONL file (train.jsonl or val.jsonl)."
    )
    parser.add_argument(
        "--numberbatch", required=False,
        help="ConceptNet Numberbatch embeddings (.txt or .txt.gz)."
    )
    parser.add_argument(
        "--assertions", default=None,
        help="ConceptNet assertions CSV.gz (optional; improves edge structure)."
    )
    parser.add_argument(
        "--output", required=False,
        help="Output HDF5 path for KG graphs."
    )
    parser.add_argument(
        "--d-kg", type=int, default=300,
        help="KG embedding dimension (default 300). Must match configs."
    )
    parser.add_argument(
        "--num-visual-nodes", type=int, default=36,
        help="Number of visual region nodes (default 36)."
    )
    parser.add_argument(
        "--max-kg-nodes", type=int, default=30,
        help="Maximum KG nodes per question (default 30)."
    )
    parser.add_argument(
        "--validate-visual", default=None,
        help="If set, validate that all img_fns in JSONL are in this HDF5."
    )
    args = parser.parse_args()

    # Validation mode
    if args.validate_visual:
        if not args.jsonl:
            parser.error("--jsonl is required with --validate-visual")
        validate_visual_features(args.jsonl, args.validate_visual)
        return

    # KG building mode
    required = ["jsonl", "numberbatch", "output"]
    missing = [r for r in required if not getattr(args, r.replace("-", "_"))]
    if missing:
        parser.error(f"The following arguments are required: {', '.join('--'+m for m in missing)}")

    # 1. Load VCR questions and convert to VQA format
    questions = load_vcr_questions(args.jsonl)

    # 2. Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp_path = tmp.name
        save_vqa_format_questions(questions, tmp_path)

    # 3. Call prepare_kg_graphs.py with the temp file
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

    print(f"\nCalling prepare_kg_graphs.py for VCR data...")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERROR] prepare_kg_graphs.py failed.", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\n[DONE] VCR KG graphs saved to: {args.output}")
    print("Next steps:")
    print("  1. Ensure visual features HDF5 is available (R2C Faster RCNN features).")
    print("  2. Run validation: python scripts/prepare_vcr_data.py \\")
    print(f"       --jsonl {args.jsonl} --validate-visual <path/to/visual_features.h5>")
    print("  3. Start training: python train.py --config-name baseline_vcr_qa \\")
    print("       datasets=vcr_qa datasets.train.data_dir=data/vcr \\")
    print("       datasets.val.data_dir=data/vcr")


if __name__ == "__main__":
    main()
