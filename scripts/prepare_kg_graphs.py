"""
Build per-question KG subgraphs from ConceptNet for VQA-GNN.

Produces HDF5 files compatible with VQADataset._load_graph.

Output format:
    data/vqa/knowledge_graphs/{split}_graphs.h5

    Key: question_id (string)
    Contents (HDF5 group):
        node_features:  float32[N_kg, d_kg]        KG node embeddings
        adj_matrix:     float32[N_total, N_total]   full adjacency matrix
        node_types:     int32[N_total]              0=visual, 1=question, 2=kg

    Where N_total = num_visual_nodes + 1 + N_kg
    Visual node positions: 0 .. num_visual_nodes-1
    Question node position: num_visual_nodes
    KG node positions:      num_visual_nodes+1 .. N_total-1

DEVIATION FROM PAPER (explicitly documented):
  1. Entity extraction: simple whitespace tokenization + stopword filtering.
     The paper likely uses NER or entity linking; this script uses a simple
     heuristic. Impact: some relevant entities may be missed.
  2. Node features: ConceptNet Numberbatch embeddings (default 300-dim).
     The paper uses custom KG embeddings whose details are not fully disclosed.
     The d_kg dimension must match src/configs/datasets/vqa.yaml and
     src/configs/model/vqa_gnn.yaml. Update both configs if using 300-dim.
  3. Edge structure: ConceptNet 5.7 assertions (if provided). Without the
     assertions file, KG nodes are connected as a fully-connected subgraph.
  4. Adjacency: visual nodes and question node are fully connected. KG nodes
     are connected to the question node. This is an assumption; the paper
     does not fully specify the adjacency structure.

Required inputs:
    1. VQA v2 question JSON file (train or val)
    2. ConceptNet Numberbatch embeddings:
       Download (English, ~450MB compressed):
         wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz

    Optional:
    3. ConceptNet assertions CSV (for richer KG-internal edge structure):
       Download (~500MB compressed):
         wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
       Without this file, KG-internal edges are fully connected (approximation).

Config alignment:
    The d_kg value passed here must match:
      src/configs/datasets/vqa.yaml        → d_kg
      src/configs/datasets/vqa_eval.yaml   → d_kg
      src/configs/model/vqa_gnn.yaml       → d_kg

    Default here is 300 (full Numberbatch). Default in configs is 100 (for demo).
    To use 300-dim features, update both the script call and the YAML configs.

Usage:
    # Build train graphs (300-dim features, needs config update):
    python scripts/prepare_kg_graphs.py \\
        --questions data/vqa/questions/train_questions.json \\
        --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
        --output data/vqa/knowledge_graphs/train_graphs.h5 \\
        --d-kg 300

    # Build val graphs:
    python scripts/prepare_kg_graphs.py \\
        --questions data/vqa/questions/val_questions.json \\
        --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
        --output data/vqa/knowledge_graphs/val_graphs.h5 \\
        --d-kg 300

    # With ConceptNet assertions (better KG edge structure):
    python scripts/prepare_kg_graphs.py \\
        --questions data/vqa/questions/train_questions.json \\
        --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
        --assertions data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \\
        --output data/vqa/knowledge_graphs/train_graphs.h5 \\
        --d-kg 300

    # With 100-dim features (matches default configs, truncates Numberbatch to 100 dims):
    python scripts/prepare_kg_graphs.py \\
        --questions data/vqa/questions/train_questions.json \\
        --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \\
        --output data/vqa/knowledge_graphs/train_graphs.h5 \\
        --d-kg 100
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("[ERROR] h5py is required. Install: pip install h5py", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_TYPE_VISUAL = 0
NODE_TYPE_QUESTION = 1
NODE_TYPE_KG = 2

# Stopwords for entity extraction (English)
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "and", "but", "or", "nor", "not",
    "so", "yet", "both", "either", "neither", "whether", "if", "as",
    "because", "while", "when", "where", "how", "what", "which", "who",
    "whom", "whose", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "any", "some", "no", "s",
    "more", "most", "other", "than", "such", "into", "only", "same",
    "very", "just", "also", "there", "here",
})

# Question words to skip (common in VQA questions but usually not good KG entities)
_QUESTION_WORDS = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "are", "is", "was", "were", "do", "does", "did", "can", "could",
    "would", "should", "will", "has", "have", "had", "does", "many",
    "much", "long", "old", "far", "often",
})


# ---------------------------------------------------------------------------
# Numberbatch loading
# ---------------------------------------------------------------------------


def load_numberbatch(path: str, target_dim: int = 300):
    """
    Load ConceptNet Numberbatch embeddings from a .txt or .txt.gz file.

    Supported file formats:
        1. English-only Numberbatch:
           <num_terms> <embedding_dim>
           cat  0.1 0.2 ... 0.3
           fire_truck  0.1 0.2 ... 0.3

        2. Full ConceptNet URI format:
           <num_terms> <embedding_dim>
           /c/en/cat  0.1 0.2 ... 0.3
           /c/en/fire_truck  0.1 0.2 ... 0.3

    Args:
        path (str): path to numberbatch-en.txt or numberbatch-en.txt.gz
        target_dim (int): desired output dimension. If target_dim < file_dim,
            features are truncated (documented approximation). If equal, used as-is.
    Returns:
        vocab (dict): {entity_str: np.ndarray[target_dim]} e.g. {"cat": array(...)}
        embedding_dim (int): actual embedding dimension in the file
    """
    path = Path(path)
    if not path.exists():
        print(
            f"[ERROR] Numberbatch file not found: {path}\n"
            "Download with:\n"
            "  wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
            "numberbatch-en-19.08.txt.gz",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading Numberbatch embeddings from: {path}")
    print("(This may take a minute for the English-only file ~450MB)")

    opener = gzip.open if str(path).endswith(".gz") else open
    vocab = {}
    embedding_dim = None
    n_total = None

    with opener(path, "rt", encoding="utf-8") as f:
        first_line = f.readline().strip().split()
        if len(first_line) == 2:
            # Header line: num_terms embedding_dim
            n_total = int(first_line[0])
            embedding_dim = int(first_line[1])
        else:
            # No header; try to infer from first data line
            try:
                parts = first_line
                # First token is either a ConceptNet URI or an english-only term.
                embedding_dim = len(parts) - 1
                entity_key = _numberbatch_key(parts[0])
                if entity_key is not None:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if target_dim < embedding_dim:
                        vec = vec[:target_dim]
                    vocab[entity_key] = vec
            except Exception:
                pass

        it = tqdm(f, total=n_total, desc="Loading Numberbatch", unit="lines")
        for line in it:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entity_key = _numberbatch_key(parts[0])
            if entity_key is None:
                continue
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if target_dim < len(vec):
                vec = vec[:target_dim]
            vocab[entity_key] = vec

    print(f"Loaded {len(vocab)} English entity embeddings (dim={embedding_dim}).")
    if target_dim < (embedding_dim or target_dim):
        print(
            f"[DEVIATION] Truncating embeddings from {embedding_dim} to {target_dim} dims. "
            "This is an approximation. Update --d-kg and your YAML configs to use "
            f"the full {embedding_dim} dimensions for better quality."
        )
    return vocab, embedding_dim


def _uri_to_key(uri: str):
    """
    Convert a ConceptNet URI to a simple English entity string.

    Examples:
        /c/en/cat      → 'cat'
        /c/en/cat/n    → 'cat'
        /c/en/hot_dog  → 'hot dog'
        /c/fr/chat     → None  (non-English)
        /r/AtLocation  → None  (relation, not entity)
    """
    if not uri.startswith("/c/en/"):
        return None
    parts = uri.split("/")
    # parts: ['', 'c', 'en', 'entity', optional_pos, ...]
    if len(parts) < 4:
        return None
    entity = parts[3]
    # Replace underscores with spaces
    entity = entity.replace("_", " ")
    return entity if entity else None


def _numberbatch_key(token: str):
    """
    Convert a Numberbatch entry key to the normalized entity string used in KG
    lookup.

    Supports both:
        - full ConceptNet URIs, e.g. /c/en/fire_truck -> 'fire truck'
        - english-only Numberbatch tokens, e.g. fire_truck -> 'fire truck'
    """
    if token.startswith("/"):
        return _uri_to_key(token)

    token = token.strip()
    if not token:
        return None
    return token.replace("_", " ")


# ---------------------------------------------------------------------------
# ConceptNet assertions loading
# ---------------------------------------------------------------------------


def load_cn_edges(assertions_path: str, vocab_set: set):
    """
    Load ConceptNet edges between English entities present in vocab_set.

    Returns:
        edges (dict): {entity: set of neighbors} for entities in vocab_set
    """
    path = Path(assertions_path)
    if not path.exists():
        print(
            f"[ERROR] Assertions file not found: {path}\n"
            "Download with:\n"
            "  wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
            "conceptnet-assertions-5.7.0.csv.gz",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading ConceptNet assertions from: {path}")
    print("(This takes several minutes for the full ~500MB file)")

    edges: dict = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    n_loaded = 0

    with opener(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading assertions", unit="lines"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            # parts[1] = relation, parts[2] = head, parts[3] = tail
            head_key = _uri_to_key(parts[2])
            tail_key = _uri_to_key(parts[3])
            if head_key is None or tail_key is None:
                continue
            if head_key not in vocab_set or tail_key not in vocab_set:
                continue
            edges.setdefault(head_key, set()).add(tail_key)
            edges.setdefault(tail_key, set()).add(head_key)
            n_loaded += 1

    print(f"Loaded {n_loaded} edges between {len(edges)} vocabulary entities.")
    return edges


# ---------------------------------------------------------------------------
# Entity extraction from question text
# ---------------------------------------------------------------------------


def extract_entities(question: str):
    """
    Extract candidate entity strings from a VQA question.

    DEVIATION FROM PAPER:
        The paper likely uses NER or entity linking. Here we use simple
        tokenization with stopword removal.

    Args:
        question (str): raw question string.
    Returns:
        entities (list[str]): candidate entity strings.
    """
    # Lowercase and clean
    text = question.lower()
    # Remove punctuation
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    tokens = text.split()

    # Remove stopwords and question words, keep content words
    entities = [
        t for t in tokens
        if t not in _STOPWORDS
        and t not in _QUESTION_WORDS
        and len(t) > 2
    ]

    # Also try bigrams (e.g. "tennis ball", "fire truck")
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)
               if tokens[i] not in _STOPWORDS and tokens[i+1] not in _STOPWORDS
               and len(tokens[i]) > 2 and len(tokens[i+1]) > 2]

    # Return unique, bigrams first (more specific)
    seen = set()
    result = []
    for e in bigrams + entities:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_question_graph(
    question: str,
    nb_vocab: dict,
    cn_edges: dict,
    max_kg_nodes: int,
    num_visual_nodes: int,
    d_kg: int,
    zero_vector: np.ndarray,
):
    """
    Build the full adjacency matrix and node features for one question.

    DEVIATION FROM PAPER: See module docstring for full deviation list.

    Returns:
        node_features (np.ndarray): float32[N_kg, d_kg]
        adj_matrix (np.ndarray): float32[N_total, N_total]
        node_types (np.ndarray): int32[N_total]

    N_total = num_visual_nodes + 1 + N_kg
    """
    entities = extract_entities(question)

    # Match entities to Numberbatch vocabulary
    matched = []
    seen_entities = set()
    for entity in entities:
        if entity in nb_vocab and entity not in seen_entities:
            matched.append(entity)
            seen_entities.add(entity)
        # Try without last character (simple singularization: cats → cat)
        elif len(entity) > 3 and entity.endswith("s") and entity[:-1] in nb_vocab:
            base = entity[:-1]
            if base not in seen_entities:
                matched.append(base)
                seen_entities.add(base)
        if len(matched) >= max_kg_nodes:
            break

    # Fallback: if no entities matched at all, use a single zero-vector node
    if not matched:
        node_features = zero_vector.reshape(1, d_kg)
        n_kg = 1
    else:
        n_kg = min(len(matched), max_kg_nodes)
        node_features = np.stack([nb_vocab[e] for e in matched[:n_kg]], axis=0)

    # N_total for this question
    n_total = num_visual_nodes + 1 + n_kg
    q_idx = num_visual_nodes  # question node index

    # Build adjacency matrix
    adj = np.zeros((n_total, n_total), dtype=np.float32)

    # Self-loops for all nodes
    np.fill_diagonal(adj, 1.0)

    # Visual ↔ question node: fully connected
    adj[:num_visual_nodes, q_idx] = 1.0
    adj[q_idx, :num_visual_nodes] = 1.0

    # Visual ↔ visual: fully connected within visual region
    # (common in VQA graph models; paper not fully specified)
    adj[:num_visual_nodes, :num_visual_nodes] = 1.0

    # KG nodes ↔ question node
    kg_start = num_visual_nodes + 1
    kg_end = n_total
    adj[kg_start:kg_end, q_idx] = 1.0
    adj[q_idx, kg_start:kg_end] = 1.0

    # KG-internal edges from ConceptNet assertions
    if cn_edges:
        for i, ent_i in enumerate(matched[:n_kg]):
            for j, ent_j in enumerate(matched[:n_kg]):
                if i == j:
                    continue
                if ent_j in cn_edges.get(ent_i, set()):
                    adj[kg_start + i, kg_start + j] = 1.0
                    adj[kg_start + j, kg_start + i] = 1.0
    else:
        # Fallback: KG nodes fully connected to each other
        # DEVIATION: fully-connected KG subgraph (paper uses ConceptNet paths)
        if n_kg > 1:
            adj[kg_start:kg_end, kg_start:kg_end] = 1.0

    # Node types
    node_types = np.array(
        [NODE_TYPE_VISUAL] * num_visual_nodes
        + [NODE_TYPE_QUESTION]
        + [NODE_TYPE_KG] * n_kg,
        dtype=np.int32,
    )

    return node_features, adj, node_types


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build per-question KG subgraphs for VQA-GNN from ConceptNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--questions",
        required=True,
        metavar="PATH",
        help="VQA v2 question JSON file (e.g. data/vqa/questions/train_questions.json).",
    )
    parser.add_argument(
        "--numberbatch",
        required=True,
        metavar="PATH",
        help=(
            "Path to ConceptNet Numberbatch .txt or .txt.gz file.\n"
            "English-only download: "
            "wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
            "numberbatch-en-19.08.txt.gz"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output HDF5 path (e.g. data/vqa/knowledge_graphs/train_graphs.h5).",
    )
    parser.add_argument(
        "--assertions",
        default=None,
        metavar="PATH",
        help=(
            "Optional: ConceptNet assertions CSV(.gz) for KG-internal edge structure.\n"
            "Without this, KG nodes are fully connected (approximation).\n"
            "Download: "
            "wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
            "conceptnet-assertions-5.7.0.csv.gz"
        ),
    )
    parser.add_argument(
        "--d-kg",
        type=int,
        default=300,
        help=(
            "KG node feature dimension. Must match d_kg in YAML configs "
            "(src/configs/model/vqa_gnn.yaml and src/configs/datasets/vqa.yaml). "
            "Default: 300 (full Numberbatch). "
            "If < 300, Numberbatch vectors are truncated (approximation). "
            "Demo configs default to 100; update them if using 300."
        ),
    )
    parser.add_argument(
        "--max-kg-nodes",
        type=int,
        default=30,
        help=(
            "Max KG nodes per question (default: 30). "
            "Must match max_kg_nodes in YAML configs."
        ),
    )
    parser.add_argument(
        "--num-visual-nodes",
        type=int,
        default=36,
        help=(
            "Number of visual region nodes (default: 36). "
            "Must match num_visual_nodes in YAML configs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N questions (for debugging).",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression (default: gzip).",
    )

    args = parser.parse_args()

    # Validate inputs
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"[ERROR] Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    # Load questions
    print(f"Loading questions from: {questions_path}")
    with open(questions_path) as f:
        q_data = json.load(f)
    questions = q_data["questions"]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Processing {len(questions)} questions.")

    # Load Numberbatch
    nb_vocab, nb_dim = load_numberbatch(args.numberbatch, target_dim=args.d_kg)

    if args.d_kg > nb_dim:
        print(
            f"[ERROR] Requested d_kg={args.d_kg} but Numberbatch has only "
            f"{nb_dim} dimensions.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load ConceptNet edges (optional)
    cn_edges = {}
    if args.assertions:
        nb_vocab_set = set(nb_vocab.keys())
        cn_edges = load_cn_edges(args.assertions, nb_vocab_set)
    else:
        print(
            "[DEVIATION] No assertions file provided. KG nodes will be fully connected. "
            "Provide --assertions for ConceptNet-based edge structure."
        )

    # Zero vector for fallback (questions with no entity matches)
    zero_vector = np.zeros(args.d_kg, dtype=np.float32)

    # Build graphs and write to HDF5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression
    n_written = 0
    n_empty = 0

    with h5py.File(output_path, "w") as hf:
        for q in tqdm(questions, desc="Building graphs"):
            qid = str(q["question_id"])
            question_text = q["question"]

            node_features, adj_matrix, node_types = build_question_graph(
                question=question_text,
                nb_vocab=nb_vocab,
                cn_edges=cn_edges,
                max_kg_nodes=args.max_kg_nodes,
                num_visual_nodes=args.num_visual_nodes,
                d_kg=args.d_kg,
                zero_vector=zero_vector,
            )

            if node_features.shape[0] == 1 and np.allclose(node_features, 0):
                n_empty += 1

            grp = hf.create_group(qid)
            grp.create_dataset(
                "node_features",
                data=node_features.astype(np.float32),
                compression=compression,
            )
            grp.create_dataset(
                "adj_matrix",
                data=adj_matrix.astype(np.float32),
                compression=compression,
            )
            grp.create_dataset(
                "node_types",
                data=node_types.astype(np.int32),
                compression=compression,
            )
            n_written += 1

    print(f"\nWritten {n_written} question graphs to: {output_path}")
    if n_empty > 0:
        print(
            f"[INFO] {n_empty} questions had no entity matches in Numberbatch; "
            "a single zero-vector KG node was used as fallback."
        )

    # Print deviation summary
    print("\n--- DEVIATION SUMMARY ---")
    print("Entity extraction: simple tokenization (paper: likely NER/entity linking)")
    print(f"Node features: Numberbatch {nb_dim}-dim, stored as {args.d_kg}-dim")
    if args.d_kg < nb_dim:
        print(f"  Truncated {nb_dim}→{args.d_kg} (approximation)")
    print(
        "Edge structure: "
        + ("ConceptNet assertions" if cn_edges else "fully-connected KG subgraph (approximation)")
    )
    print("Adjacency: visual↔question + kg↔question + kg↔kg from above")
    print("Visual↔visual: fully connected (paper not fully specified)")
    print("--- END DEVIATION SUMMARY ---")
    print(
        f"\nReminder: update src/configs/model/vqa_gnn.yaml (d_kg: {args.d_kg}) and "
        f"src/configs/datasets/vqa.yaml (d_kg: {args.d_kg}) to match output dimension."
    )


if __name__ == "__main__":
    main()
