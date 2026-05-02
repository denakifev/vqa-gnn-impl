"""
GQA (Generalized Visual Question Answering) dataset for VQA-GNN (arXiv:2205.11501).

Two classes are provided:

GQADataset
    Reads real GQA balanced-split data (JSON files from cs.stanford.edu/people/dorarad/gqa).
    Requires pre-extracted visual features (HDF5, keyed by imageId),
    a pre-built answer vocabulary JSON, and pre-built per-question scene-graph
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
    graph_node_features     FloatTensor[N_kg, d_kg]       textual SG node features
    graph_adj               FloatTensor[N_total, N_total] adjacency matrix
    graph_edge_types        LongTensor[N_total, N_total]  relation ids
    graph_node_types        LongTensor[N_total]           0=visual,1=q,2=kg
    labels                  LongTensor[]                  GT answer index
    question_id             str                           GQA question ID

Where N_total = N_v + 1 + N_kg.

GQA differences from VQA v2:
    - Single ground-truth answer per question (no soft labels from multiple annotators).
    - Paper target answer vocabulary size: 1842 classes, not 3129 (VQA v2).
    - Images from Visual Genome.
    - Annotation JSON format: {q_id: {question, imageId, answer, ...}}.

Deviation from paper:
    - Scene-graph tensors are pre-built offline rather than on-the-fly.
    - Graph edges are stored as dense adjacency / relation-id matrices.
    - The fixed answer space is inferred from the released balanced splits rather
      than copied from an official paper artifact.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.gqa_preprocessing import normalize_answer

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
            │                                    #   {node_features, adj_matrix, graph_edge_types, node_types}
            └── val_graphs.h5

        answer_vocab_path points to a JSON file:
        {
          "answer_to_idx": {"yes": 0, "no": 1, ...},
          "idx_to_answer": ["yes", "no", ...],
          "metadata": {...}
        }
    Build it with: python scripts/prepare_gqa_data.py answer-vocab ...

    GQA annotation format:
        Each question JSON file is a dict: {question_id: {question, imageId, answer, ...}}

        Deviation from paper:
        Scene-graph tensors are pre-built offline rather than built on-the-fly.
        Visual features are expected in the official GQA object-feature layout
        repackaged into runtime HDF5 files keyed by imageId.
    """

    def __init__(
        self,
        partition: str,
        data_dir: str,
        answer_vocab_path: str,
        max_question_len: int = 30,
        num_visual_nodes: int = 100,
        max_kg_nodes: int = 100,
        d_visual: int = 2048,
        d_kg: int = 600,
        text_encoder_name: str = "roberta-large",
        instance_transforms=None,
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        strict_answer_vocab: bool = True,
        expect_graph_edge_types: bool = True,
        strict_graph_schema: bool = True,
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
            strict_answer_vocab (bool): fail if a labeled sample cannot be
                mapped into the provided answer vocabulary.
            expect_graph_edge_types (bool): if True, require `graph_edge_types`
                to be present in every graph HDF5 group. Set to False to
                accept legacy artifacts produced by an older preprocessing
                path that did not store typed edges (edge types will be
                zero-filled). This is a data-contract toggle, not a silent
                fallback — an explicit config decision is required.
            strict_graph_schema (bool): if True, probe the first graph HDF5
                sample on first HDF5 open and fail fast when its
                `node_features` width does not equal `d_kg`, or when
                `graph_edge_types` is missing while
                `expect_graph_edge_types=True`. See DATA_CONTRACTS.md.
        """
        self.partition = partition
        self.data_dir = Path(data_dir)
        self.max_question_len = max_question_len
        self.num_visual_nodes = num_visual_nodes
        self.max_kg_nodes = max_kg_nodes
        self.d_visual = d_visual
        self.d_kg = d_kg
        self.n_total = num_visual_nodes + 1 + max_kg_nodes
        self.strict_answer_vocab = strict_answer_vocab
        self.expect_graph_edge_types = expect_graph_edge_types
        self.strict_graph_schema = strict_graph_schema

        # Load answer vocabulary
        vocab_path = Path(answer_vocab_path)
        _check_file_exists(vocab_path, "GQA answer vocabulary")
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        self.answer_to_idx: dict = vocab_data["answer_to_idx"]
        self.idx_to_answer: list[str] = vocab_data.get("idx_to_answer", [])
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
        unknown_answers = []
        for q_id, item in q_data.items():
            answer = item.get("answer", None)
            normalized_answer = normalize_answer(answer) if answer else None
            label = self.answer_to_idx.get(normalized_answer, -1) if normalized_answer else -1
            if answer and label < 0 and self.partition != "test":
                if len(unknown_answers) < 10:
                    unknown_answers.append(
                        {
                            "question_id": str(q_id),
                            "raw_answer": answer,
                            "normalized_answer": normalized_answer,
                        }
                    )
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

        if unknown_answers and self.strict_answer_vocab:
            raise ValueError(
                "Found labeled GQA samples whose answers are missing from the answer vocabulary. "
                "Rebuild the vocab with the official balanced splits or inspect normalization. "
                f"Examples: {unknown_answers}"
            )
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
            self._validate_h5_metadata()
            if self.strict_graph_schema:
                self._validate_graph_sample_contract()

    def _validate_h5_metadata(self) -> None:
        """
        Validate optional file-level attrs written by the new GQA pipeline.

        Old artifacts without attrs remain loadable; new artifacts get stricter
        consistency checks to catch data/config mismatches early. A dedicated
        sample-level probe (`_validate_graph_sample_contract`) covers legacy
        artifacts that lack these attrs altogether.
        """
        visual_num_boxes = self._visual_h5.attrs.get("num_boxes")
        visual_feature_dim = self._visual_h5.attrs.get("feature_dim")
        if visual_num_boxes is not None and int(visual_num_boxes) != self.num_visual_nodes:
            raise ValueError(
                f"[GQA data contract] Visual HDF5 declares num_boxes={visual_num_boxes}, "
                f"but dataset config requests num_visual_nodes={self.num_visual_nodes}.\n"
                f"  file: {self._visual_h5_path}\n"
                f"  fix: align dataset config `num_visual_nodes` or rebuild visual "
                f"features with: scripts/prepare_gqa_data.py visual-features "
                f"--num-boxes {self.num_visual_nodes}"
            )
        if visual_feature_dim is not None and int(visual_feature_dim) != self.d_visual:
            raise ValueError(
                f"[GQA data contract] Visual HDF5 declares feature_dim={visual_feature_dim}, "
                f"but dataset config requests d_visual={self.d_visual}.\n"
                f"  file: {self._visual_h5_path}"
            )

        graph_d_kg = self._graph_h5.attrs.get("d_kg")
        graph_num_visual_nodes = self._graph_h5.attrs.get("num_visual_nodes")
        graph_max_kg_nodes = self._graph_h5.attrs.get("max_kg_nodes")
        if graph_d_kg is not None and int(graph_d_kg) != self.d_kg:
            raise ValueError(
                f"[GQA data contract] Graph HDF5 declares d_kg={graph_d_kg}, "
                f"but dataset config requests d_kg={self.d_kg}.\n"
                f"  file: {self._graph_h5_path}\n"
                f"  fix: set model/dataset `d_kg={graph_d_kg}` or rebuild graphs with: "
                f"scripts/prepare_gqa_data.py knowledge-graphs ... --d-kg {self.d_kg}"
            )
        if (
            graph_num_visual_nodes is not None
            and int(graph_num_visual_nodes) != self.num_visual_nodes
        ):
            raise ValueError(
                f"[GQA data contract] Graph HDF5 declares num_visual_nodes="
                f"{graph_num_visual_nodes}, but dataset config requests "
                f"num_visual_nodes={self.num_visual_nodes}.\n"
                f"  file: {self._graph_h5_path}"
            )
        if graph_max_kg_nodes is not None and int(graph_max_kg_nodes) != self.max_kg_nodes:
            raise ValueError(
                f"[GQA data contract] Graph HDF5 declares max_kg_nodes="
                f"{graph_max_kg_nodes}, but dataset config requests "
                f"max_kg_nodes={self.max_kg_nodes}.\n"
                f"  file: {self._graph_h5_path}"
            )

        if self._graph_h5.attrs.get("fully_connected_fallback_used") is True:
            logger.warning(
                "KG HDF5 reports fully_connected_fallback_used=True. "
                "This artifact predates the stricter paper-aligned GQA pipeline."
            )

    def _validate_graph_sample_contract(self) -> None:
        """
        Probe the first graph HDF5 entry and fail fast on contract violations.

        Legacy artifacts (older GQA pipeline) are missing the file-level attrs
        that `_validate_h5_metadata` checks. This sample-level probe catches
        the common mismatches regardless of whether attrs are present:

          - `node_features` dim != d_kg             (e.g. 300 vs expected 600)
          - `graph_edge_types` missing while `expect_graph_edge_types=True`
          - required datasets missing (`adj_matrix`, `node_features`, ...)

        Errors include: what was expected, what was found, the file path, and
        a short remediation hint. See DATA_CONTRACTS.md.
        """
        sample_key: Optional[str] = None
        for entry in self._index:
            qid = entry["question_id"]
            if qid in self._graph_h5:
                sample_key = qid
                break
        if sample_key is None:
            # Fall back to first group in HDF5 if no index entry matches (rare).
            keys = list(self._graph_h5.keys())
            if not keys:
                raise ValueError(
                    f"[GQA data contract] Graph HDF5 has no entries.\n"
                    f"  file: {self._graph_h5_path}"
                )
            sample_key = keys[0]

        grp = self._graph_h5[sample_key]

        for required in ("node_features", "adj_matrix", "node_types"):
            if required not in grp:
                raise ValueError(
                    f"[GQA data contract] Graph HDF5 group '{sample_key}' is missing "
                    f"required dataset '{required}'.\n"
                    f"  file: {self._graph_h5_path}\n"
                    f"  expected group members: node_features, adj_matrix, "
                    f"node_types, graph_edge_types"
                )

        actual_d_kg = int(grp["node_features"].shape[-1])
        if actual_d_kg != self.d_kg:
            raise ValueError(
                f"[GQA data contract] Graph HDF5 sample '{sample_key}' has "
                f"node_features with d_kg={actual_d_kg}, but dataset config "
                f"requests d_kg={self.d_kg}.\n"
                f"  file: {self._graph_h5_path}\n"
                f"  fix options (pick one):\n"
                f"    (a) rebuild graphs with the current scene-graph pipeline:\n"
                f"        python scripts/prepare_gqa_data.py knowledge-graphs ... "
                f"--d-kg {self.d_kg}\n"
                f"    (b) align config to the stored artifact — override both\n"
                f"        model.d_kg={actual_d_kg} and datasets.train.d_kg="
                f"{actual_d_kg} datasets.val.d_kg={actual_d_kg}\n"
                f"        (non paper-faithful; see DATA_CONTRACTS.md).\n"
                f"  blocking a silent dim mismatch mid-batch is intentional."
            )

        has_edge_types = "graph_edge_types" in grp
        if self.expect_graph_edge_types and not has_edge_types:
            raise ValueError(
                f"[GQA data contract] Graph HDF5 sample '{sample_key}' is missing "
                f"`graph_edge_types`, but `expect_graph_edge_types=True`.\n"
                f"  file: {self._graph_h5_path}\n"
                f"  fix options (pick one):\n"
                f"    (a) rebuild graphs with the current pipeline to emit typed\n"
                f"        edges:  python scripts/prepare_gqa_data.py "
                f"knowledge-graphs ...\n"
                f"    (b) accept legacy untyped artifacts explicitly by setting\n"
                f"        datasets.train.expect_graph_edge_types=False "
                f"datasets.val.expect_graph_edge_types=False\n"
                f"        (edge types will be zero-filled; `not paper-faithful yet`)."
            )
        if not self.expect_graph_edge_types and not has_edge_types:
            logger.warning(
                "[GQA data contract] Graph HDF5 artifacts do not carry "
                "`graph_edge_types`. Running with zero-filled edge types "
                "per explicit `expect_graph_edge_types=False`. This run is "
                "`not paper-faithful yet` with respect to GQA scene-graph "
                "relations."
            )

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
        graph_node_features, graph_adj, graph_edge_types, graph_node_types = self._load_graph(
            question_id
        )

        if entry["label"] < 0 and self.partition != "test" and self.strict_answer_vocab:
            raise ValueError(
                f"Question {question_id} has no valid answer label in the current answer vocab."
            )

        label = torch.tensor(max(entry["label"], 0), dtype=torch.long)

        return {
            "visual_features": visual_features,
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "graph_node_features": graph_node_features,
            "graph_adj": graph_adj,
            "graph_edge_types": graph_edge_types,
            "graph_node_types": graph_node_types,
            "labels": label,
            "question_id": question_id,
        }

    def _load_graph(self, question_id: str) -> tuple:
        """Load a pre-built GQA scene-graph sample for the given question."""
        grp = self._graph_h5[question_id]

        node_feats_raw = torch.tensor(grp["node_features"][()], dtype=torch.float32)
        n_kg_actual = node_feats_raw.shape[0]
        if n_kg_actual > self.max_kg_nodes:
            node_feats_raw = node_feats_raw[: self.max_kg_nodes]
            n_kg_actual = self.max_kg_nodes
        graph_node_features = torch.zeros(self.max_kg_nodes, self.d_kg)
        graph_node_features[:n_kg_actual] = node_feats_raw

        graph_adj = torch.tensor(grp["adj_matrix"][()], dtype=torch.float32)
        if "graph_edge_types" in grp:
            graph_edge_types = torch.tensor(grp["graph_edge_types"][()], dtype=torch.long)
        else:
            graph_edge_types = torch.zeros_like(graph_adj, dtype=torch.long)
        graph_node_types = torch.tensor(grp["node_types"][()], dtype=torch.long)

        n_total_expected = self.num_visual_nodes + 1 + self.max_kg_nodes
        graph_adj = _pad_or_trim_2d(graph_adj, n_total_expected)
        graph_edge_types = _pad_or_trim_2d(graph_edge_types, n_total_expected)

        cur_nt_len = graph_node_types.shape[0]
        if cur_nt_len > n_total_expected:
            graph_node_types = graph_node_types[:n_total_expected]
        elif cur_nt_len < n_total_expected:
            extra = n_total_expected - cur_nt_len
            graph_node_types = torch.cat(
                [graph_node_types, torch.full((extra,), NODE_TYPE_KG, dtype=torch.long)]
            )

        return graph_node_features, graph_adj, graph_edge_types, graph_node_types


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
        num_answers: int = 1842,
        max_question_len: int = 30,
        num_visual_nodes: int = 100,
        max_kg_nodes: int = 100,
        d_visual: int = 2048,
        d_kg: int = 600,
        seed: int = 42,
        instance_transforms=None,
        **kwargs,
    ):
        """
        Args:
            num_samples (int): number of synthetic samples.
            num_answers (int): GQA answer vocabulary size (default 1842, paper target).
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
        Nv = self.num_visual_nodes
        Nt = self.max_kg_nodes
        q_idx = Nv  # convention from gqa_preprocessing.py: q after visual block

        visual_features = torch.tensor(
            rng.standard_normal((Nv, self.d_visual)).astype(np.float32)
        )
        question_input_ids = torch.randint(0, 50264, (self.max_question_len,))
        question_attention_mask = torch.ones(self.max_question_len, dtype=torch.long)
        graph_node_features = torch.tensor(
            rng.standard_normal((Nt, self.d_kg)).astype(np.float32)
        )

        # Build adjacency matching the preprocessing invariant:
        # visual↔textual cross-edges are zero (only q connects them).
        adj_np = np.zeros((n_total, n_total), dtype=np.float32)
        # Visual block: random sparse symmetric
        v_block = (rng.random((Nv, Nv)) > 0.7).astype(np.float32)
        v_block = np.maximum(v_block, v_block.T)
        adj_np[:Nv, :Nv] = v_block
        # Textual block: random sparse symmetric
        t_block = (rng.random((Nt, Nt)) > 0.7).astype(np.float32)
        t_block = np.maximum(t_block, t_block.T)
        adj_np[Nv + 1 :, Nv + 1 :] = t_block
        # Q row/col: q connected to all visual + all textual (paper §4.1)
        adj_np[q_idx, :Nv] = 1.0
        adj_np[:Nv, q_idx] = 1.0
        adj_np[q_idx, Nv + 1 :] = 1.0
        adj_np[Nv + 1 :, q_idx] = 1.0
        # Self loops
        np.fill_diagonal(adj_np, 1.0)
        graph_adj = torch.tensor(adj_np)

        # Edge types respecting the same block structure.
        edge_types_np = np.zeros((n_total, n_total), dtype=np.int64)
        # Random ids on each block; values bounded to avoid overflowing demo
        # `num_relations` upper bounds. These are demo synthetic values, not
        # real relation ids.
        v_rel = rng.integers(2, 8, size=(Nv, Nv)).astype(np.int64)
        edge_types_np[:Nv, :Nv] = np.where(adj_np[:Nv, :Nv] > 0, v_rel, 0)
        t_rel = rng.integers(2, 8, size=(Nt, Nt)).astype(np.int64)
        edge_types_np[Nv + 1 :, Nv + 1 :] = np.where(adj_np[Nv + 1 :, Nv + 1 :] > 0, t_rel, 0)
        # Q-visual / q-textual special types
        edge_types_np[q_idx, :Nv] = 2  # __question_visual__ id placeholder
        edge_types_np[:Nv, q_idx] = 2
        edge_types_np[q_idx, Nv + 1 :] = 3  # __question_textual__ id placeholder
        edge_types_np[Nv + 1 :, q_idx] = 3
        np.fill_diagonal(edge_types_np, 1)  # __self__ id placeholder
        graph_edge_types = torch.tensor(edge_types_np, dtype=torch.long)

        node_types = np.array(
            [NODE_TYPE_VISUAL] * Nv
            + [NODE_TYPE_QUESTION]
            + [NODE_TYPE_KG] * Nt,
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
            "graph_edge_types": graph_edge_types,
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

