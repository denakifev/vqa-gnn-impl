"""
VCR (Visual Commonsense Reasoning) dataset for VQA-GNN (arXiv:2205.11501).

Two classes are provided:

VCRDataset
    Reads real VCR data (train/val/test JSONL files from allenai/vcr).
    Requires pre-extracted visual features (HDF5, keyed by img_fn),
    and pre-built per-question KG subgraph files (HDF5, keyed by annot_id).
    See README.md → "Data preparation (VCR)" for required directory layout.

VCRDemoDataset
    Generates fully synthetic VCR-like data for smoke testing and CI.
    DOES NOT reproduce any result from the paper.
    Field shapes and dtypes match VCRDataset so the same model, loss, metrics,
    collate, and trainer code is exercised.

Task modes:
    'qa'  — Q→A: encode [question; candidate_answer_i] as QA-context node.
             Model scored each of the 4 answer candidates.
    'qar' — QA→R: encode [question; correct_answer; candidate_rationale_j] as
             QA-context node. Model scores each of the 4 rationale candidates.

Batch contract (keys returned by __getitem__ for both classes):
    visual_features         FloatTensor[N_v, d_visual]   region features
    qa_input_ids            LongTensor[4, L]             Q+A_i encodings
    qa_attention_mask       LongTensor[4, L]             text encoder masks
    qar_input_ids           LongTensor[4, L]             Q+A+R_j encodings
    qar_attention_mask      LongTensor[4, L]             text encoder masks
    graph_node_features     FloatTensor[N_kg, d_kg]      KG node features
    graph_adj               FloatTensor[N_total, N_total] adjacency matrix
    graph_node_types        LongTensor[N_total]          0=visual,1=q,2=kg
    answer_label            LongTensor[]                 GT answer index 0–3
    rationale_label         LongTensor[]                 GT rationale index 0–3
    annot_id                str                          VCR annotation ID

Where N_total = N_v + 1 + N_kg.

Deviation from paper:
    - KG subgraphs are pre-built offline (paper queries ConceptNet at runtime).
    - VCR object references (e.g., [0, 2]) are resolved to category name strings.
      The paper may use object-region grounding; exact details are not disclosed.
    - KG is built from question text only (same KG for all 4 candidates).
      The QA-context diversity comes from the text encoder, not the KG.
    - Visual features assumed to be 36×2048 bottom-up attention features.
      Official VCR uses R2C Faster RCNN features (also ~2048-dim, variable count).
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
    "[DEMO MODE — VCR] Running with synthetic data. "
    "This is NOT a paper experiment. Use real VCR data for research."
)

# Number of answer/rationale candidates in VCR (always 4)
VCR_NUM_CANDIDATES = 4


def _vcr_tokens_to_text(tokens: list, objects: Optional[list] = None) -> str:
    """
    Convert a VCR token list to a plain text string.

    VCR tokens can be strings or integer lists (object references).
    Object references like [0, 2] mean "the 0th and 2nd detected objects".
    We replace them with the object category name(s) if `objects` is provided.

    Args:
        tokens (list): VCR token list from annotations.
        objects (list[str] | None): list of object category names for this image.
    Returns:
        text (str): plain text string.
    """
    parts = []
    for tok in tokens:
        if isinstance(tok, list):
            # Object reference: list of integer indices
            if objects:
                names = [objects[i] for i in tok if i < len(objects)]
                parts.append(" and ".join(names) if names else "object")
            else:
                parts.append("object")
        else:
            parts.append(str(tok))
    return " ".join(parts)


def _check_file_exists(path: Path, description: str) -> None:
    """Raise FileNotFoundError with a helpful message if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            "Please prepare the data according to README.md → 'Data preparation (VCR)'."
        )


# ---------------------------------------------------------------------------
# Real VCR Dataset
# ---------------------------------------------------------------------------


class VCRDataset(Dataset):
    """
    VCR dataset for VQA-GNN training and evaluation.

    Required directory layout (all paths relative to data_dir):

        data_dir/
        ├── train.jsonl                    # VCR annotation file
        ├── val.jsonl
        ├── test.jsonl                     # no labels
        ├── visual_features/
        │   ├── train_features.h5          # HDF5: img_fn → float32[N_v, 2048]
        │   └── val_features.h5
        └── knowledge_graphs/
            ├── train_graphs.h5            # HDF5: annot_id → {node_features,
            │                              #   adj_matrix, node_types}
            └── val_graphs.h5

    task_mode controls which QA-context encoding the model uses:
        'qa'  → Q→A: encode [question; answer_candidate_i]
        'qar' → QA→R: encode [question; correct_answer; rationale_candidate_j]

    Deviation from paper:
        VCR object references are resolved to category name text.
        KG is built from question text only (same for all 4 candidates).
        Visual features assumed to be 36×2048 bottom-up style features.
    """

    def __init__(
        self,
        partition: str,
        data_dir: str,
        task_mode: str = "qa",
        max_context_len: int = 64,
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
            data_dir (str): root directory with VCR data (see layout above).
            task_mode (str): 'qa' for Q→A, 'qar' for QA→R.
            max_context_len (int): max token length for QA context.
            num_visual_nodes (int): number of visual region features.
            max_kg_nodes (int): max number of KG nodes per sample.
            d_visual (int): visual feature dimension.
            d_kg (int): KG node feature dimension.
            text_encoder_name (str): Hugging Face tokenizer identifier.
            instance_transforms: unused; kept for API compatibility.
            limit (int|None): if set, use only first `limit` samples.
            shuffle_index (bool): shuffle samples before applying limit.
        """
        if task_mode not in ("qa", "qar"):
            raise ValueError(f"task_mode must be 'qa' or 'qar', got: {task_mode!r}")

        self.partition = partition
        self.data_dir = Path(data_dir)
        self.task_mode = task_mode
        self.max_context_len = max_context_len
        self.num_visual_nodes = num_visual_nodes
        self.max_kg_nodes = max_kg_nodes
        self.d_visual = d_visual
        self.d_kg = d_kg
        self.n_total = num_visual_nodes + 1 + max_kg_nodes

        # Load Hugging Face tokenizer
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        except ImportError as e:
            raise ImportError(
                "transformers is required for VCRDataset. "
                "Install: pip install transformers"
            ) from e

        # Build index from JSONL
        logger.info(f"Building VCRDataset index [{partition}, mode={task_mode}]...")
        self._index = self._build_index(limit if not shuffle_index else None)

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
        _check_file_exists(self._visual_h5_path, f"VCR visual features ({partition})")
        _check_file_exists(self._graph_h5_path, f"VCR knowledge graphs ({partition})")

        logger.info(
            f"VCRDataset [{partition}, mode={task_mode}]: {len(self._index)} samples."
        )

        from src.datasets.vcr_collate import make_vcr_collate_fn

        self.collate_fn = make_vcr_collate_fn(task_mode=task_mode)

    def _build_index(self, limit: Optional[int] = None) -> list:
        """Parse VCR JSONL file and build the sample index."""
        jsonl_path = self.data_dir / f"{self.partition}.jsonl"
        _check_file_exists(jsonl_path, f"VCR annotations ({self.partition})")

        index = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                index.append(
                    {
                        "annot_id": item["annot_id"],
                        "img_fn": item["img_fn"],
                        "question": item["question"],
                        "answer_choices": item["answer_choices"],
                        "answer_label": item.get("answer_label", -1),
                        "rationale_choices": item.get("rationale_choices", []),
                        "rationale_label": item.get("rationale_label", -1),
                        "objects": item.get("objects", []),
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
                    "h5py is required for VCRDataset. Install: pip install h5py"
                ) from e
            self._visual_h5 = h5py.File(self._visual_h5_path, "r")
            self._graph_h5 = h5py.File(self._graph_h5_path, "r")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        self._open_h5()
        entry = self._index[idx]
        objects = entry["objects"]
        annot_id = entry["annot_id"]
        img_fn = entry["img_fn"]

        # Visual features [N_v, d_visual]
        visual_features = torch.tensor(
            self._visual_h5[img_fn][()], dtype=torch.float32
        )
        if visual_features.shape[0] != self.num_visual_nodes:
            visual_features = _pad_or_trim(visual_features, self.num_visual_nodes)

        # Tokenize QA context for each of the 4 answer candidates (Q→A)
        q_text = _vcr_tokens_to_text(entry["question"], objects)
        qa_input_ids, qa_attention_mask = self._encode_candidates(
            q_text, entry["answer_choices"], objects
        )

        # Tokenize QAR context for each of the 4 rationale candidates (QA→R)
        # Uses the ground-truth answer for the correct answer slot.
        # At test time, the predicted answer from the Q→A model is used instead,
        # but during training the GT answer is used (standard VCR protocol).
        correct_answer_idx = max(entry["answer_label"], 0)  # fallback 0 for test
        correct_a_text = _vcr_tokens_to_text(
            entry["answer_choices"][correct_answer_idx]
            if entry["answer_choices"]
            else [],
            objects,
        )
        if entry["rationale_choices"]:
            qar_input_ids, qar_attention_mask = self._encode_candidates_qar(
                q_text, correct_a_text, entry["rationale_choices"], objects
            )
        else:
            # test split: no rationale choices — fill with zeros
            qar_input_ids = torch.zeros(VCR_NUM_CANDIDATES, self.max_context_len, dtype=torch.long)
            qar_attention_mask = torch.zeros(VCR_NUM_CANDIDATES, self.max_context_len, dtype=torch.long)

        # KG subgraph (shared for all 4 candidates)
        graph_node_features, graph_adj, graph_node_types = self._load_graph(annot_id)

        answer_label = torch.tensor(max(entry["answer_label"], 0), dtype=torch.long)
        rationale_label = torch.tensor(max(entry["rationale_label"], 0), dtype=torch.long)

        return {
            "visual_features": visual_features,
            "qa_input_ids": qa_input_ids,
            "qa_attention_mask": qa_attention_mask,
            "qar_input_ids": qar_input_ids,
            "qar_attention_mask": qar_attention_mask,
            "graph_node_features": graph_node_features,
            "graph_adj": graph_adj,
            "graph_node_types": graph_node_types,
            "answer_label": answer_label,
            "rationale_label": rationale_label,
            "annot_id": annot_id,
        }

    def _encode_candidates(
        self, q_text: str, candidates: list, objects: list
    ) -> tuple:
        """
        Tokenize [question; candidate_answer_i] for each of the 4 candidates.

        Returns:
            input_ids (LongTensor[4, L])
            attention_mask (LongTensor[4, L])
        """
        all_ids = []
        all_masks = []
        for cand in candidates[:VCR_NUM_CANDIDATES]:
            a_text = _vcr_tokens_to_text(cand, objects)
            enc = self.tokenizer(
                q_text,
                a_text,
                max_length=self.max_context_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            all_ids.append(enc["input_ids"].squeeze(0))
            all_masks.append(enc["attention_mask"].squeeze(0))
        # Pad to 4 candidates if fewer provided (e.g., corrupted test data)
        while len(all_ids) < VCR_NUM_CANDIDATES:
            all_ids.append(torch.zeros(self.max_context_len, dtype=torch.long))
            all_masks.append(torch.zeros(self.max_context_len, dtype=torch.long))
        return torch.stack(all_ids), torch.stack(all_masks)

    def _encode_candidates_qar(
        self, q_text: str, a_text: str, rationale_candidates: list, objects: list
    ) -> tuple:
        """
        Tokenize [question; answer; candidate_rationale_j] for each of the 4 rationale candidates.

        Returns:
            input_ids (LongTensor[4, L])
            attention_mask (LongTensor[4, L])
        """
        all_ids = []
        all_masks = []
        qa_prefix = q_text + " " + a_text
        for cand in rationale_candidates[:VCR_NUM_CANDIDATES]:
            r_text = _vcr_tokens_to_text(cand, objects)
            enc = self.tokenizer(
                qa_prefix,
                r_text,
                max_length=self.max_context_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            all_ids.append(enc["input_ids"].squeeze(0))
            all_masks.append(enc["attention_mask"].squeeze(0))
        while len(all_ids) < VCR_NUM_CANDIDATES:
            all_ids.append(torch.zeros(self.max_context_len, dtype=torch.long))
            all_masks.append(torch.zeros(self.max_context_len, dtype=torch.long))
        return torch.stack(all_ids), torch.stack(all_masks)

    def _load_graph(self, annot_id: str) -> tuple:
        """
        Load pre-built KG subgraph for a given VCR annotation.

        Expected HDF5 structure (same as VQADataset._load_graph):
            node_features: float32[N_kg_actual, d_kg]
            adj_matrix:    float32[N_total, N_total]
            node_types:    int32[N_total]

        Returns:
            graph_node_features (FloatTensor[max_kg_nodes, d_kg])
            graph_adj (FloatTensor[N_total, N_total])
            graph_node_types (LongTensor[N_total])
        """
        grp = self._graph_h5[annot_id]

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


class VCRDemoDataset(Dataset):
    """
    Synthetic VCR-like dataset for smoke testing.

    WARNING: Does NOT reproduce any VCR experiment.
    Generates random tensors matching VCRDataset field shapes.
    """

    def __init__(
        self,
        task_mode: str = "qa",
        num_samples: int = 64,
        max_context_len: int = 64,
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
            task_mode (str): 'qa' or 'qar' — controls collate behaviour only.
            num_samples (int): number of synthetic samples.
            max_context_len (int): token sequence length.
            num_visual_nodes (int): number of visual region nodes.
            max_kg_nodes (int): number of KG nodes per sample.
            d_visual (int): visual feature dimension.
            d_kg (int): KG node feature dimension.
            seed (int): random seed.
        """
        logger.warning(_DEMO_NOTICE)
        if task_mode not in ("qa", "qar"):
            raise ValueError(f"task_mode must be 'qa' or 'qar', got: {task_mode!r}")
        self.task_mode = task_mode
        self.num_samples = num_samples
        self.max_context_len = max_context_len
        self.num_visual_nodes = num_visual_nodes
        self.max_kg_nodes = max_kg_nodes
        self.d_visual = d_visual
        self.d_kg = d_kg
        self.n_total = num_visual_nodes + 1 + max_kg_nodes

        rng = np.random.default_rng(seed)
        self._data = [self._gen_sample(rng, i) for i in range(num_samples)]

        from src.datasets.vcr_collate import make_vcr_collate_fn

        self.collate_fn = make_vcr_collate_fn(task_mode=task_mode)

    def _gen_sample(self, rng: np.random.Generator, idx: int) -> dict:
        n_total = self.n_total
        L = self.max_context_len

        visual_features = torch.tensor(
            rng.standard_normal((self.num_visual_nodes, self.d_visual)).astype(np.float32)
        )

        # 4 QA context encodings
        qa_input_ids = torch.randint(0, 50264, (VCR_NUM_CANDIDATES, L))
        qa_attention_mask = torch.ones(VCR_NUM_CANDIDATES, L, dtype=torch.long)
        qar_input_ids = torch.randint(0, 50264, (VCR_NUM_CANDIDATES, L))
        qar_attention_mask = torch.ones(VCR_NUM_CANDIDATES, L, dtype=torch.long)

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

        answer_label = torch.tensor(int(rng.integers(0, VCR_NUM_CANDIDATES)), dtype=torch.long)
        rationale_label = torch.tensor(int(rng.integers(0, VCR_NUM_CANDIDATES)), dtype=torch.long)

        return {
            "visual_features": visual_features,
            "qa_input_ids": qa_input_ids,
            "qa_attention_mask": qa_attention_mask,
            "qar_input_ids": qar_input_ids,
            "qar_attention_mask": qar_attention_mask,
            "graph_node_features": graph_node_features,
            "graph_adj": graph_adj,
            "graph_node_types": graph_node_types,
            "answer_label": answer_label,
            "rationale_label": rationale_label,
            "annot_id": f"demo-{idx}",
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
