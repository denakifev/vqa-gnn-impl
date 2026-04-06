"""
Build answer vocabulary from VQA v2 training annotations.

Output format:
    { "answer_to_idx": { "yes": 0, "no": 1, ... } }

The vocabulary includes the top-N most frequent normalized answers from the
training annotation file. Ties in count are broken lexicographically to ensure
determinism across runs and platforms.

Answer normalization follows the standard VQA preprocessing used in most
VQA v2 baselines (adapted from the official vqa-tools eval code):
  - Lowercase
  - Strip leading/trailing whitespace
  - Remove trailing period (unless between digits)
  - Remove common punctuation (;/[]()"?)
  - Handle number contractions (two -> 2, etc.)

DEVIATION FROM PAPER:
    The paper does not explicitly document its normalization pipeline.
    The normalization here follows the widely-used VQA v2 baseline convention
    (Anderson et al., CVPR 2018; Antol et al., ICCV 2015 eval script).
    Exact vocabulary membership may differ slightly from the paper's.

Usage:
    python scripts/prepare_answer_vocab.py \\
        --annotations data/vqa/annotations/train_annotations.json \\
        --output data/vqa/answer_vocab.json \\
        --top-n 3129

    # With multiple annotation files (e.g. train + val combined):
    python scripts/prepare_answer_vocab.py \\
        --annotations data/vqa/annotations/train_annotations.json \\
                      data/vqa/annotations/val_annotations.json \\
        --output data/vqa/answer_vocab.json \\
        --top-n 3129
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Answer normalization (VQA v2 standard, based on official vqa-tools)
# ---------------------------------------------------------------------------

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT_PATTERN = re.compile(r"[;/\[\]()\"]")
_PUNCT_PATTERN2 = re.compile(r"(\?|\!|\\|\+|\^|\$|\@|\#|\%|\&|\*|\{|\}|\||\:|\.)")

# VQA number word normalization (spelled-out → digits)
# Covers the most common cases appearing in VQA v2 answers.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}

# Contractions that commonly appear in VQA v2 answers
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "lets": "let's",
    "mightve": "might've",
    "mustve": "must've",
    "mustnt": "mustn't",
    "neednt": "needn't",
    "notve": "not've",
    "oughtnt": "oughtn't",
    "shes": "she's",
    "shed": "she'd",
    "shell": "she'll",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "thatll": "that'll",
    "thats": "that's",
    "thered": "there'd",
    "thereve": "there've",
    "theyve": "they've",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "wasnt": "wasn't",
    "weve": "we've",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


def normalize_answer(answer: str) -> str:
    """
    Normalize a VQA answer string.

    Applies the standard VQA v2 preprocessing pipeline:
    lowercase → strip → number-word substitution → contraction restoration
    → comma/period strip → punctuation removal.

    Args:
        answer (str): raw answer string.
    Returns:
        normalized (str): normalized answer string.
    """
    answer = answer.lower().strip()

    # Number words → digits
    tokens = answer.split()
    tokens = [_NUMBER_WORDS.get(tok, tok) for tok in tokens]
    answer = " ".join(tokens)

    # Contractions
    tokens = answer.split()
    tokens = [_CONTRACTIONS.get(tok, tok) for tok in tokens]
    answer = " ".join(tokens)

    # Comma between digits: 1,000 → 1000
    answer = _COMMA_STRIP.sub(r"\1\3", answer)

    # Period not between digits → remove
    answer = _PERIOD_STRIP.sub("", answer)

    # Common punctuation → space
    answer = _PUNCT_PATTERN.sub(" ", answer)
    answer = _PUNCT_PATTERN2.sub("", answer)

    answer = answer.strip()
    return answer


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------


def build_vocab(annotation_paths: list, top_n: int) -> dict:
    """
    Count normalized answers across all annotation files and return the top-N.

    Each annotation has 10 human answers. All 10 are counted independently.
    Ties are broken lexicographically (ascending) for determinism.

    Args:
        annotation_paths (list[str]): paths to VQA v2 annotation JSON files.
        top_n (int): vocabulary size.
    Returns:
        answer_to_idx (dict): mapping from normalized answer string to index.
    """
    counter: Counter = Counter()

    for ann_path in annotation_paths:
        ann_path = Path(ann_path)
        if not ann_path.exists():
            print(f"[ERROR] Annotation file not found: {ann_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading annotations from: {ann_path}")
        with open(ann_path) as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        if not annotations:
            print(
                f"[WARNING] No annotations found in {ann_path}. "
                "Expected key 'annotations' in JSON root.",
                file=sys.stderr,
            )

        for ann in annotations:
            for a in ann.get("answers", []):
                raw = a.get("answer", "").strip()
                if raw:
                    normalized = normalize_answer(raw)
                    if normalized:
                        counter[normalized] += 1

    print(f"Total unique normalized answers: {len(counter)}")
    print(f"Total answer occurrences: {sum(counter.values())}")

    # Sort: (count descending, answer ascending for ties)
    sorted_answers = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    if top_n > len(sorted_answers):
        print(
            f"[WARNING] Requested top-{top_n} but only {len(sorted_answers)} "
            "unique answers found. Using all answers.",
            file=sys.stderr,
        )
        top_n = len(sorted_answers)

    top_answers = [ans for ans, _ in sorted_answers[:top_n]]
    answer_to_idx = {ans: idx for idx, ans in enumerate(top_answers)}

    # Sanity check: coverage
    total_anns = sum(counter.values())
    covered = sum(counter[ans] for ans in top_answers)
    coverage_pct = 100.0 * covered / total_anns if total_anns > 0 else 0.0
    print(
        f"Vocabulary of {len(answer_to_idx)} answers covers "
        f"{coverage_pct:.1f}% of all answer occurrences."
    )

    return answer_to_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build answer vocabulary from VQA v2 training annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "Path(s) to VQA v2 annotation JSON file(s). "
            "Usually: data/vqa/annotations/train_annotations.json"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for answer_vocab.json (e.g. data/vqa/answer_vocab.json).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3129,
        help=(
            "Number of most frequent answers to include. "
            "VQA v2 standard is 3129 (default)."
        ),
    )
    parser.add_argument(
        "--no-number-norm",
        action="store_true",
        help="Disable number-word normalization (two -> 2 etc.).",
    )

    args = parser.parse_args()

    if args.no_number_norm:
        _NUMBER_WORDS.clear()

    answer_to_idx = build_vocab(args.annotations, args.top_n)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_data = {"answer_to_idx": answer_to_idx}
    with open(output_path, "w") as f:
        json.dump(vocab_data, f, ensure_ascii=False)

    print(f"Answer vocabulary saved to: {output_path}")
    print(f"Vocabulary size: {len(answer_to_idx)}")
    print("Top 10 answers by frequency index:")
    for ans, idx in list(answer_to_idx.items())[:10]:
        print(f"  [{idx:4d}] {ans}")


if __name__ == "__main__":
    main()
