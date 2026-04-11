"""
GQA (Generalized Visual Question Answering) dataset for VQA-GNN (arXiv:2205.11501).

Two classes are provided:

GQADataset
    Reads real GQA balanced-split data (JSON files from cs.stanford.edu/people/dorarad/gqa).
    Requires pre-extracted visual features (HDF5, keyed by imageId),
    a pre-built answer vocabulary JSON, and pre-built per-question KG subgraph
    files (HDF5, keyed by question_id).
    See README.md → "Data preparation (GQA)" for required directory layout.

GQADemoDataset
    Generates fully synthetic GQA-like data for smoke testing and CI.
    DOES NOT reproduce any result from the paper.
    Field shapes and dtypes match GQADataset so the same model, loss, metrics,
    collate, and trainer code is exercised.

Batch contract (keys returned by __getitem__ for both classes):
    visual_features         FloatTensor[N_v, d_visual]    region features
    question_input_ids      LongTensor[L]                 tokenized question
    question_attention_mask LongTensor[L]                 text encoder mask
    graph_node_features     FloatTensor[N_kg, d_kg]       KG node features
    graph_adj               FloatTensor[N_total, N_total] adjacency matrix
    graph_node_types        LongTensor[N_total]           0=visual,1=q,2=kg
    labels                  LongTensor[]                  GT answer index
    question_id             str                           GQA question ID

Where N_total = N_v + 1 + N_kg.

GQA differences from VQA v2:
    - Single ground-truth answer per question (no soft labels from multiple annotators).
    - Answer vocabulary size ~1852 (GQA balanced split), not 3129 (VQA v2).
    - Images from Visual Genome (same dataset as VCR).
    - Annotation JSON format: {q_id: {question, imageId, answer, ...}}.

Deviation from paper:
    - KG subgraphs are pre-built offline (paper queries ConceptNet at runtime).
    - Visual features assumed to be bottom-up style (paper uses GQA's own features).
    - Exact answer vocabulary mapping follows a top-N frequency count on train set.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Node type constants — must match VQAGNNModel
NODE_TYPE_VISUAL = 0
NODE_TYPE_QUESTION = 1
NODE_TYPE_KG = 2

_DEMO_NOTICE = (
    "[DEMO MODE — GQA] Running with synthetic data. "
    "This is NOT a paper experiment. Use real GQA data for research."
)


def _check_file_exists(path: Path, description: str) -> None:
    """Raise FileNotFoundError with a helpful message if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            "Please prepare the data according to README.md → 'Data preparation (GQA)'."
        )


# ---------------------------------------------------------------------------
# Real GQA Dataset
# ---------------------------------------------------------------------------


class GQADataset(Dataset):
    """
    GQA balanced-split dataset for VQA-GNN training and evaluation.

    Required directory layout (all paths relative to data_dir):

        data_dir/
        ├── questions/
        │   ├── train_balanced_questions.json    # GQA balanced train questions
        │   └── val_balanced_questions.json       # GQA balanced val questions
        ├── visual_features/
        │   ├── train_features.h5               # HDF5: imageId (str) → float32[N_v, d_visual]
        │   └── val_features.h5
        └── knowledge_graphs/
            ├── train_graphs.h5                  # HDF5: question_id (str) →
            │                                    #   {node_features, adj_matrix, node_types}
            └── val_graphs.h5

    answer_vocab_path points to a JSON file:
        { "answer_to_idx": {"yes": 0, "no": 1, ...} }
    Build it with: python scripts/prepare_gqa_data.py --build-vocab ...

    GQA annotation format:
        Each question JSON file is a dict: {question_id: {question, imageId, answer, ...}}

    Deviation from paper:
        KG subgraphs are pre-built offline rather than queried at runtime.
        Visual features are expected as pre-extracted bottom-up style features.
        GQA's own features (and detectors) are not used here — this is a
        documented approximation; the exact visual feature source should match
        across VCR and GQA for a paper-aligned joint experiment.
    """

    def __init__(
        self,
        partition: str,
        data_dir: str,
        answer_vocab_path: str,
        max_question_len: int = 30,
        num_visual_nodes: int = 36,
        max_kg_nodes: int = 30,
        d_visual: int = 2048,
        d_kg: int = 300,
        text_encoder_name: str = "roberta-large",
        instance_transforms=None,
        limit: Optional[int] = None,
        shuffle_index: bool = False,
    ):
        """
        Args:
            partition (str): one of "train", "val", "test".
            data_dir (str): root directory with GQA data (see layout above).
            answer_vocab_path (str): path to answer vocabulary JSON.
            max_question_len (int): max question token length.
            num_visual_nodes (int): number of visual region features.
            max_kg_nodes (int): max number of KG nodes per sample.
            d_visual (int): visual feature dimension.
            d_kg (int): KG node feature dimension.
            text_encoder_name (str): Hugging Face tokenizer identifier.
            instance_transforms: unused; kept for API compatibility.
            limit (int|None): if set, use only first `limit` samples.
            shuffle_index (bool): shuffle samples before applying limit.
        """
        self.partition = partition
        self.data_dir = Path(data_dir)
        self.max_question_len = max_question_len
        self.num_visual_nodes = num_visual_nodes
        self.max_kg_nodes = max_kg_nodes
        self.d_visual = d_visual
        self.d_kg = d_kg
        self.n_total = num_visual_nodes + 1 + max_kg_nodes

        # Load answer vocabulary
        vocab_path = Path(answer_vocab_path)
        _check_file_exists(vocab_path, "GQA answer vocabulary")
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        self.answer_to_idx: dict = vocab_data["answer_to_idx"]
        self.num_answers = len(self.answer_to_idx)
        logger.info(f"Loaded GQA answer vocabulary: {self.num_answers} answers.")

        # Load Hugging Face tokenizer
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        except ImportError as e:
            raise ImportError(
                "transformers is required for GQADataset. "
                "Install: pip install transformers"
            ) from e

        # Build index from JSON files
        logger.info(f"Building GQADataset index [{partition}]...")
        early_limit = limit if (limit is not None and not shuffle_index) else None
        self._index = self._build_index(limit=early_limit)

        if shuffle_index:
            random.seed(42)
            random.shuffle(self._index)
        if limit is not None:
            self._index = self._index[:limit]

        # Lazy HDF5 handles
        self._visual_h5 = None
        self._graph_h5 = None
        self._visual_h5_path = self.data_dir / "visual_features" / f"{partition}_features.h5"
        self._graph_h5_path = self.data_dir / "knowledge_graphs" / f"{partition}_graphs.h5"
        _check_file_exists(self._visual_h5_path, f"GQA visual features ({partition})")
        _check_file_exists(self._graph_h5_path, f"GQA knowledge graphs ({partition})")

        logger.info(f"GQADataset [{partition}]: {len(self._index)} samples.")

        from src.datasets.gqa_collate import gqa_collate_fn

        self.collate_fn = gqa_collate_fn

    def _build_index(self, limit: Optional[int] = None) -> list:
        """
        Build list of sample metadata from GQA JSON question file.

        GQA question file format:
            {
                "00001": {
                    "question": "Is the sky blue?",
                    "imageId": "2375429",
                    "answer": "yes",
                    "fullAnswer": "Yes, the sky is blue.",
                    ...
                },
                ...
            }
        """
        q_file = f"{self.partition}_balanced_questions.json"
        q_path = self.data_dir / "questions" / q_file
        _check_file_exists(q_path, f"GQA questions ({self.partition})")

        with open(q_path) as f:
            q_data = json.load(f)

        index = []
        for q_id, item in q_data.items():
            answer = item.get("answer", None)
            label = self.answer_to_idx.get(answer, -1) if answer else -1
            index.append(
                {
                    "question_id": q_id,
                    "image_id": str(item["imageId"]),
                    "question": item["question"],
                    "label": label,
                }
            )
            if limit is not None and len(index) >= limit:
                break
        return index

    def _open_h5(self):
        """Lazily open HDF5 files (safe for multiprocessing workers)."""
        if self._visual_h5 is None:
            try:
                import h5py
            except ImportError as e:
                raise ImportError(
                    "h5py is required for GQADataset. Install: pip install h5py"
                ) from e
            self._visual_h5 = h5py.File(self._visual_h5_path, "r")
            self._graph_h5 = h5py.File(self._graph_h5_path, "r")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        self._open_h5()
        entry = self._index[idx]
        image_id = entry["image_id"]
        question_id = entry["question_id"]

        # Visual features [N_v, d_visual]
        visual_features = torch.tensor(
            self._visual_h5[image_id][()], dtype=torch.float32
        )
        if visual_features.shape[0] != self.num_visual_nodes:
            visual_features = _pad_or_trim(visual_features, self.num_visual_nodes)

        # Tokenize question
        enc = self.tokenizer(
            entry["question"],
            max_length=self.max_question_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        question_input_ids = enc["input_ids"].squeeze(0)
        question_attention_mask = enc["attention_mask"].squeeze(0)

        # KG subgraph
        graph_node_features, graph_adj, graph_node_types = self._load_graph(question_id)

        label = torch.tensor(max(entry["label"], 0), dtype=torch.long)

        return {
            "visual_features": visual_features,
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "graph_node_features": graph_node_features,
            "graph_adj": graph_adj,
            "graph_node_types": graph_node_types,
            "labels": label,
            "question_id": question_id,
        }

    def _load_graph(self, question_id: str) -> tuple:
        """Load pre-built KG subgraph for a given GQA question (same format as VQADataset)."""
        grp = self._graph_h5[question_id]

        node_feats_raw = torch.tensor(grp["node_features"][()], dtype=torch.float32)
        n_kg_actual = node_feats_raw.shape[0]
        if n_kg_actual > self.max_kg_nodes:
            node_feats_raw = node_feats_raw[: self.max_kg_nodes]
            n_kg_actual = self.max_kg_nodes
        graph_node_features = torch.zeros(self.max_kg_nodes, self.d_kg)
        graph_node_features[:n_kg_actual] = node_feats_raw

        graph_adj = torch.tensor(grp["adj_matrix"][()], dtype=torch.float32)
        graph_node_types = torch.tensor(grp["node_types"][()], dtype=torch.long)

        n_total_expected = self.num_visual_nodes + 1 + self.max_kg_nodes
        graph_adj = _pad_or_trim_2d(graph_adj, n_total_expected)

        cur_nt_len = graph_node_types.shape[0]
        if cur_nt_len > n_total_expected:
            graph_node_types = graph_node_types[:n_total_expected]
        elif cur_nt_len < n_total_expected:
            extra = n_total_expected - cur_nt_len
            graph_node_types = torch.cat(
                [graph_node_types, torch.full((extra,), NODE_TYPE_KG, dtype=torch.long)]
            )

        return graph_node_features, graph_adj, graph_node_types


# ---------------------------------------------------------------------------
# Demo Dataset (synthetic, for smoke testing only)
# ---------------------------------------------------------------------------


class GQADemoDataset(Dataset):
    """
    Synthetic GQA-like dataset for smoke testing.

    WARNING: Does NOT reproduce any GQA experiment.
    Generates random tensors matching GQADataset field shapes.
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_answers: int = 1852,
        max_question_len: int = 30,
        num_visual_nodes: int = 36,
        max_kg_nodes: int = 30,
        d_visual: int = 2048,
        d_kg: int = 100,
        seed: int = 42,
        instance_transforms=None,
        **kwargs,
    ):
        """
        Args:
            num_samples (int): number of synthetic samples.
            num_answers (int): GQA answer vocabulary size (default 1852 balanced split).
            max_question_len (int): question token sequence length.
            num_visual_nodes (int): number of visual region nodes.
            max_kg_nodes (int): number of KG nodes per sample.
            d_visual (int): visual feature dimension.
            d_kg (int): KG node feature dimension.
            seed (int): random seed.
        """
        logger.warning(_DEMO_NOTICE)
        self.num_samples = num_samples
        self.num_answers = num_answers
        self.max_question_len = max_question_len
        self.num_visual_nodes = num_visual_nodes
        self.max_kg_nodes = max_kg_nodes
        self.d_visual = d_visual
        self.d_kg = d_kg
        self.n_total = num_visual_nodes + 1 + max_kg_nodes

        rng = np.random.default_rng(seed)
        self._data = [self._gen_sample(rng, i) for i in range(num_samples)]

        from src.datasets.gqa_collate import gqa_collate_fn

        self.collate_fn = gqa_collate_fn

    def _gen_sample(self, rng: np.random.Generator, idx: int) -> dict:
        n_total = self.n_total

        visual_features = torch.tensor(
            rng.standard_normal((self.num_visual_nodes, self.d_visual)).astype(np.float32)
        )
        question_input_ids = torch.randint(0, 50264, (self.max_question_len,))
        question_attention_mask = torch.ones(self.max_question_len, dtype=torch.long)
        graph_node_features = torch.tensor(
            rng.standard_normal((self.max_kg_nodes, self.d_kg)).astype(np.float32)
        )
        adj_np = (rng.random((n_total, n_total)) > 0.7).astype(np.float32)
        np.fill_diagonal(adj_np, 1.0)
        graph_adj = torch.tensor(adj_np)
        node_types = np.array(
            [NODE_TYPE_VISUAL] * self.num_visual_nodes
            + [NODE_TYPE_QUESTION]
            + [NODE_TYPE_KG] * self.max_kg_nodes,
            dtype=np.int64,
        )
        graph_node_types = torch.tensor(node_types, dtype=torch.long)
        label = torch.tensor(int(rng.integers(0, self.num_answers)), dtype=torch.long)

        return {
            "visual_features": visual_features,
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "graph_node_features": graph_node_features,
            "graph_adj": graph_adj,
            "graph_node_types": graph_node_types,
            "labels": label,
            "question_id": str(idx),
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pad_or_trim(t: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = t.shape[0]
    if cur == target_len:
        return t
    if cur > target_len:
        return t[:target_len]
    pad_shape = list(t.shape)
    pad_shape[0] = target_len - cur
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)])


def _pad_or_trim_2d(t: torch.Tensor, size: int) -> torch.Tensor:
    cur = t.shape[0]
    if cur == size:
        return t
    if cur > size:
        return t[:size, :size]
    padded = torch.zeros(size, size, dtype=t.dtype)
    padded[:cur, :cur] = t
    return padded
