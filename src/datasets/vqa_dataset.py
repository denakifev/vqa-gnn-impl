"""
VQA dataset interfaces for VQA-GNN (arXiv:2205.11501).

Two classes are provided:

VQADataset
    Reads real VQA v2 data. Requires pre-extracted visual features (HDF5),
    VQA v2 question/annotation JSON files, a pre-built answer vocabulary JSON,
    and pre-built per-question knowledge-graph subgraph files (HDF5).
    See README.md → "Data preparation" for the required directory layout.

VQADemoDataset
    Generates fully synthetic VQA-like data for smoke testing and CI.
    DOES NOT reproduce any result from the paper.
    All field shapes and dtypes match VQADataset so the same model, loss,
    metrics, collate, and trainer code is exercised.

Batch contract (keys returned by __getitem__ for both classes):
    visual_features (FloatTensor[N_v, d_visual]):  region features.
    question_input_ids (LongTensor[L]):             tokenized question.
    question_attention_mask (LongTensor[L]):        text encoder attention mask.
    graph_node_features (FloatTensor[N_kg, d_kg]): KG-only node features.
    graph_adj (FloatTensor[N_total, N_total]):      adjacency matrix.
    graph_node_types (LongTensor[N_total]):         0=visual,1=q,2=kg.
    answer_scores (FloatTensor[num_answers]):       soft VQA labels.
    labels (LongTensor[]):                          hard answer index.
    question_id (int):                              VQA question id.

Where N_total = N_v + 1 + N_kg.

Deviation from paper:
    The paper builds the KG subgraph on-the-fly using ConceptNet queries
    at runtime. This implementation expects pre-built HDF5 graph files to
    allow reproducible, offline-friendly training.
    See README.md → "Knowledge graph preprocessing" for preparation steps.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer normalization (MUST stay in sync with scripts/prepare_answer_vocab.py)
# ---------------------------------------------------------------------------
# If you change the normalization here, change it in prepare_answer_vocab.py too,
# and rebuild the answer_vocab.json from scratch.

_ANS_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_ANS_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_ANS_PUNCT = re.compile(r"[;/\[\]()\"]")
_ANS_PUNCT2 = re.compile(r"(\?|\!|\\|\+|\^|\$|\@|\#|\%|\&|\*|\{|\}|\||\:|\.)")

_ANS_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ANS_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id": "i'd", "ill": "i'll", "im": "i'm", "ive": "i've", "isnt": "isn't",
    "itd": "it'd", "itll": "it'll", "its": "it's", "lets": "let's",
    "mightve": "might've", "mustve": "must've", "mustnt": "mustn't",
    "neednt": "needn't", "notve": "not've", "oughtnt": "oughtn't",
    "shes": "she's", "shed": "she'd", "shell": "she'll", "shouldve": "should've",
    "shouldnt": "shouldn't", "thatll": "that'll", "thats": "that's",
    "thered": "there'd", "thereve": "there've", "theyve": "they've",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're",
    "wasnt": "wasn't", "weve": "we've", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "youd": "you'd", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}


def _normalize_vqa_answer(answer: str) -> str:
    """
    Normalize a VQA answer string for vocabulary lookup.

    Applies the standard VQA v2 preprocessing pipeline identical to
    scripts/prepare_answer_vocab.py. Both must stay in sync so that
    answer strings produced at dataset load time match the vocabulary keys.

    Args:
        answer (str): raw answer string from VQA v2 annotations.
    Returns:
        normalized (str): normalized answer string.
    """
    answer = answer.lower().strip()
    tokens = answer.split()
    tokens = [_ANS_NUMBER_WORDS.get(tok, tok) for tok in tokens]
    answer = " ".join(tokens)
    tokens = answer.split()
    tokens = [_ANS_CONTRACTIONS.get(tok, tok) for tok in tokens]
    answer = " ".join(tokens)
    answer = _ANS_COMMA_STRIP.sub(r"\1\3", answer)
    answer = _ANS_PERIOD_STRIP.sub("", answer)
    answer = _ANS_PUNCT.sub(" ", answer)
    answer = _ANS_PUNCT2.sub("", answer)
    return answer.strip()


# Node type constants — must match VQAGNNModel
NODE_TYPE_VISUAL = 0
NODE_TYPE_QUESTION = 1
NODE_TYPE_KG = 2

_DEMO_NOTICE = (
    "[DEMO MODE] Running with synthetic data. "
    "This is NOT a paper experiment. Use real VQA v2 data for research."
)


# ---------------------------------------------------------------------------
# Real VQA v2 Dataset
# ---------------------------------------------------------------------------


class VQADataset(Dataset):
    """
    VQA v2 dataset for VQA-GNN training and evaluation.

    Required directory layout (all paths relative to data_dir):

        data_dir/
        ├── questions/
        │   ├── train_questions.json       # VQA v2 question file
        │   └── val_questions.json
        ├── annotations/
        │   ├── train_annotations.json     # VQA v2 annotation file (train only)
        │   └── val_annotations.json
        ├── visual_features/
        │   ├── train_features.h5          # HDF5: image_id (str) → float32[36, 2048]
        │   └── val_features.h5
        └── knowledge_graphs/
            ├── train_graphs.h5            # HDF5: question_id (str) →
            │                              #   node_features: float32[N_kg, d_kg]
            │                              #   adj_matrix:    float32[N_total, N_total]
            │                              #   node_types:    int32[N_total]
            └── val_graphs.h5

    answer_vocab_path points to a JSON file:
        { "answer_to_idx": {"yes": 0, "no": 1, ...} }
    This vocabulary must be built from training annotations beforehand.
    See README.md → "Data preparation".

    Deviation from paper:
        KG subgraphs are pre-built offline rather than queried at runtime.
        Visual features are expected as pre-extracted 36×2048 bottom-up
        attention features (Anderson et al., CVPR 2018). The paper may use
        different feature extractors; this is documented as an assumption.
    """

    def __init__(
        self,
        partition: str,
        data_dir: str,
        answer_vocab_path: str,
        max_question_len: int = 20,
        num_visual_nodes: int = 36,
        max_kg_nodes: int = 30,
        d_visual: int = 2048,
        d_kg: int = 100,
        text_encoder_name: str = "roberta-large",
        bert_model_name: Optional[str] = None,
        instance_transforms=None,
        limit: Optional[int] = None,
        shuffle_index: bool = False,
    ):
        """
        Args:
            partition (str): one of "train", "val", "test".
            data_dir (str): root directory with VQA data (see layout above).
            answer_vocab_path (str): path to answer vocabulary JSON.
            max_question_len (int): max question token length.
            num_visual_nodes (int): number of visual region features.
            max_kg_nodes (int): max number of KG nodes per sample.
            d_visual (int): visual feature dimension.
            d_kg (int): KG node feature dimension.
            text_encoder_name (str): Hugging Face tokenizer identifier.
            bert_model_name (str | None): deprecated alias for
                `text_encoder_name`, kept for backward compatibility.
            instance_transforms: unused; kept for API compatibility.
            limit (int|None): if set, use only first `limit` samples.
            shuffle_index (bool): shuffle samples before limit.
        """
        self.partition = partition
        self.data_dir = Path(data_dir)
        self.max_question_len = max_question_len
        self.num_visual_nodes = num_visual_nodes
        self.max_kg_nodes = max_kg_nodes
        self.d_visual = d_visual
        self.d_kg = d_kg
        self.n_total = num_visual_nodes + 1 + max_kg_nodes  # visual+q+kg

        if bert_model_name is not None:
            text_encoder_name = bert_model_name

        # Load answer vocabulary
        vocab_path = Path(answer_vocab_path)
        _check_file_exists(vocab_path, "Answer vocabulary")
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        self.answer_to_idx: dict = vocab_data["answer_to_idx"]
        self.num_answers = len(self.answer_to_idx)
        logger.info(f"Loaded answer vocabulary: {self.num_answers} answers.")

        # Load Hugging Face tokenizer
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        except ImportError as e:
            raise ImportError(
                "transformers is required for VQADataset. "
                "Install: pip install transformers"
            ) from e

        # Build index from JSON files
        self._index = self._build_index()

        if shuffle_index:
            random.seed(42)
            random.shuffle(self._index)
        if limit is not None:
            self._index = self._index[:limit]

        # Open HDF5 files lazily (opened on first __getitem__ call)
        self._visual_h5 = None
        self._graph_h5 = None
        self._visual_h5_path = self.data_dir / "visual_features" / f"{partition}_features.h5"
        self._graph_h5_path = self.data_dir / "knowledge_graphs" / f"{partition}_graphs.h5"
        _check_file_exists(self._visual_h5_path, f"Visual features ({partition})")
        _check_file_exists(self._graph_h5_path, f"Knowledge graphs ({partition})")

        logger.info(f"VQADataset [{partition}]: {len(self._index)} samples.")

        # collate_fn attribute: picked up by data_utils.get_dataloaders
        from src.datasets.vqa_collate import vqa_collate_fn

        self.collate_fn = vqa_collate_fn

    def _build_index(self) -> list:
        """Build list of dicts from question and annotation JSON files."""
        q_path = self.data_dir / "questions" / f"{self.partition}_questions.json"
        _check_file_exists(q_path, f"Questions ({self.partition})")
        with open(q_path) as f:
            q_data = json.load(f)

        # Map question_id → question text
        q_map = {item["question_id"]: item for item in q_data["questions"]}

        # For test partition there are no annotations
        if self.partition == "test":
            index = []
            for item in q_data["questions"]:
                index.append(
                    {
                        "question_id": item["question_id"],
                        "image_id": item["image_id"],
                        "question": item["question"],
                        "answer_scores": None,  # no labels for test
                        "label": -1,
                    }
                )
            return index

        ann_path = self.data_dir / "annotations" / f"{self.partition}_annotations.json"
        _check_file_exists(ann_path, f"Annotations ({self.partition})")
        with open(ann_path) as f:
            ann_data = json.load(f)

        index = []
        for ann in ann_data["annotations"]:
            qid = ann["question_id"]
            if qid not in q_map:
                continue
            q_item = q_map[qid]
            scores = _build_answer_scores(ann["answers"], self.answer_to_idx)
            label = int(scores.argmax())
            index.append(
                {
                    "question_id": qid,
                    "image_id": q_item["image_id"],
                    "question": q_item["question"],
                    "answer_scores": scores.tolist(),
                    "label": label,
                }
            )
        return index

    def _open_h5(self):
        """Lazily open HDF5 files (safe for multiprocessing workers)."""
        if self._visual_h5 is None:
            try:
                import h5py
            except ImportError as e:
                raise ImportError(
                    "h5py is required for VQADataset. Install: pip install h5py"
                ) from e
            self._visual_h5 = h5py.File(self._visual_h5_path, "r")
            self._graph_h5 = h5py.File(self._graph_h5_path, "r")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        self._open_h5()
        entry = self._index[idx]
        image_id = str(entry["image_id"])
        question_id = str(entry["question_id"])

        # Visual features [N_v, d_visual]
        visual_features = torch.tensor(
            self._visual_h5[image_id][()], dtype=torch.float32
        )
        if visual_features.shape[0] != self.num_visual_nodes:
            # Pad or truncate to num_visual_nodes
            visual_features = _pad_or_trim(visual_features, self.num_visual_nodes)

        # Tokenize question
        enc = self.tokenizer(
            entry["question"],
            max_length=self.max_question_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        question_input_ids = enc["input_ids"].squeeze(0)  # [L]
        question_attention_mask = enc["attention_mask"].squeeze(0)  # [L]

        # KG subgraph
        graph_node_features, graph_adj, graph_node_types = self._load_graph(
            question_id
        )

        # Answer labels
        if entry["answer_scores"] is not None:
            answer_scores = torch.tensor(entry["answer_scores"], dtype=torch.float32)
        else:
            answer_scores = torch.zeros(self.num_answers, dtype=torch.float32)
        label = torch.tensor(entry["label"], dtype=torch.long)

        return {
            "visual_features": visual_features,
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "graph_node_features": graph_node_features,
            "graph_adj": graph_adj,
            "graph_node_types": graph_node_types,
            "answer_scores": answer_scores,
            "labels": label,
            "question_id": entry["question_id"],
        }

    def _load_graph(self, question_id: str):
        """
        Load pre-built KG subgraph for a given question.

        Expected HDF5 structure for each question_id key:
            node_features: float32[N_kg_actual, d_kg]  (raw KG node features)
            adj_matrix:    float32[N_total, N_total]   (full adj including visual+q+kg)
            node_types:    int32[N_total]               (0=vis, 1=q, 2=kg)

        Returns:
            graph_node_features (FloatTensor[max_kg_nodes, d_kg])
            graph_adj (FloatTensor[N_total, N_total]) with N_total = N_v+1+N_kg
            graph_node_types (LongTensor[N_total])
        """
        grp = self._graph_h5[question_id]

        # KG node features
        node_feats_raw = torch.tensor(grp["node_features"][()], dtype=torch.float32)
        n_kg_actual = node_feats_raw.shape[0]
        if n_kg_actual > self.max_kg_nodes:
            node_feats_raw = node_feats_raw[: self.max_kg_nodes]
            n_kg_actual = self.max_kg_nodes
        graph_node_features = torch.zeros(self.max_kg_nodes, self.d_kg)
        graph_node_features[:n_kg_actual] = node_feats_raw

        # Full adjacency and node types (may have different N_kg in stored file)
        graph_adj = torch.tensor(grp["adj_matrix"][()], dtype=torch.float32)
        graph_node_types = torch.tensor(grp["node_types"][()], dtype=torch.long)

        # Ensure correct shape [N_total, N_total] by padding/trimming.
        n_total_expected = self.num_visual_nodes + 1 + self.max_kg_nodes
        graph_adj = _pad_or_trim_2d(graph_adj, n_total_expected)

        # Pad node_types with NODE_TYPE_KG (not zeros) for padded KG positions.
        # _pad_or_trim pads with zeros (= NODE_TYPE_VISUAL), which is wrong here.
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


class VQADemoDataset(Dataset):
    """
    Synthetic VQA-like dataset for smoke testing.

    WARNING: This dataset does NOT reproduce any VQA experiment.
    It generates random tensors with the same shapes as VQADataset
    to allow validating the full training pipeline without real data.

    All field shapes, dtypes, and keys match VQADataset.__getitem__.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_answers: int = 3129,
        max_question_len: int = 20,
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
            num_samples (int): number of synthetic samples to generate.
            num_answers (int): answer vocabulary size.
            max_question_len (int): question token sequence length.
            num_visual_nodes (int): number of visual region nodes.
            max_kg_nodes (int): number of KG nodes per sample.
            d_visual (int): visual feature dimension.
            d_kg (int): KG node feature dimension.
            seed (int): random seed for reproducibility.
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

        from src.datasets.vqa_collate import vqa_collate_fn

        self.collate_fn = vqa_collate_fn

    def _gen_sample(self, rng, idx: int) -> dict:
        n_total = self.n_total

        visual_features = torch.tensor(
            rng.standard_normal((self.num_visual_nodes, self.d_visual)).astype(np.float32)
        )
        question_input_ids = torch.randint(0, 30522, (self.max_question_len,))
        question_attention_mask = torch.ones(self.max_question_len, dtype=torch.long)
        graph_node_features = torch.tensor(
            rng.standard_normal((self.max_kg_nodes, self.d_kg)).astype(np.float32)
        )

        # Random sparse adjacency matrix with self-loops
        adj_np = (rng.random((n_total, n_total)) > 0.7).astype(np.float32)
        np.fill_diagonal(adj_np, 1.0)
        graph_adj = torch.tensor(adj_np)

        # Node types: first N_v visual, then 1 question, then N_kg kg
        node_types = np.array(
            [NODE_TYPE_VISUAL] * self.num_visual_nodes
            + [NODE_TYPE_QUESTION]
            + [NODE_TYPE_KG] * self.max_kg_nodes,
            dtype=np.int64,
        )
        graph_node_types = torch.tensor(node_types, dtype=torch.long)

        # Soft answer scores (random dirichlet-like)
        raw_scores = rng.dirichlet(np.ones(self.num_answers) * 0.1)
        answer_scores = torch.tensor(raw_scores.astype(np.float32))
        label = torch.tensor(int(answer_scores.argmax()), dtype=torch.long)

        return {
            "visual_features": visual_features,
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "graph_node_features": graph_node_features,
            "graph_adj": graph_adj,
            "graph_node_types": graph_node_types,
            "answer_scores": answer_scores,
            "labels": label,
            "question_id": idx,
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _check_file_exists(path: Path, description: str) -> None:
    """Raise FileNotFoundError with a helpful message if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            "Please prepare the data according to README.md → 'Data preparation'."
        )


def _build_answer_scores(
    answers: list, answer_to_idx: dict
) -> np.ndarray:
    """
    Build soft answer score vector from VQA v2 annotations.

    VQA v2 scoring: score = min(count_of_answer / 3, 1.0).
    Only answers that appear in the vocabulary are counted.

    Answer normalization uses _normalize_vqa_answer, which must be identical
    to the normalization in scripts/prepare_answer_vocab.py. If the two
    normalizations diverge, vocabulary lookups will silently miss answers.

    Args:
        answers (list[dict]): list of {"answer": str} dicts (10 per question).
        answer_to_idx (dict): mapping from normalized answer string to index.
    Returns:
        scores (np.ndarray): float32 array of shape [num_answers].
    """
    num_answers = len(answer_to_idx)
    counts = np.zeros(num_answers, dtype=np.float32)
    for a in answers:
        ans = _normalize_vqa_answer(a["answer"])
        if ans in answer_to_idx:
            counts[answer_to_idx[ans]] += 1.0
    scores = np.minimum(counts / 3.0, 1.0)
    return scores


def _pad_or_trim(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or trim a 1-D or 2-D tensor along dim=0 to target_len."""
    cur = t.shape[0]
    if cur == target_len:
        return t
    if cur > target_len:
        return t[:target_len]
    # Pad with zeros
    pad_shape = list(t.shape)
    pad_shape[0] = target_len - cur
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)])


def _pad_or_trim_2d(t: torch.Tensor, size: int) -> torch.Tensor:
    """Pad or trim a square 2-D tensor to [size, size]."""
    cur = t.shape[0]
    if cur == size:
        return t
    if cur > size:
        return t[:size, :size]
    # Pad with zeros on both dims
    padded = torch.zeros(size, size, dtype=t.dtype)
    padded[:cur, :cur] = t
    return padded
