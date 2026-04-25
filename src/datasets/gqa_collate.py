"""
Collate function for GQA batches (VQA-GNN, arXiv:2205.11501).

GQA is an open-ended question answering task with a fixed answer vocabulary
(paper target: 1842 answers). Each item has a single ground-truth
answer label (no soft annotations).

This collate is structurally simpler than VCR: no candidate expansion,
just standard tensor stacking.
"""

import torch


PAPER_TENSOR_KEYS = (
    "question_input_ids",
    "question_attention_mask",
    "visual_node_features",
    "visual_adj",
    "visual_edge_types",
    "visual_node_types",
    "visual_node_is_q_index",
    "textual_node_features",
    "textual_adj",
    "textual_edge_types",
    "textual_node_types",
    "textual_node_is_q_index",
    "labels",
)


def gqa_paper_collate_fn(dataset_items: list) -> dict:
    """
    Collate items emitted by `GQADataset(emit_two_subgraphs=True)` /
    `GQADemoDataset(emit_two_subgraphs=True)` into the batch format
    consumed by `src.model.PaperGQAModel`.

    Each per-item tensor field has a fixed shape (the slicer pads visual
    and textual subgraphs to `num_visual_nodes + 1` and `max_kg_nodes + 1`
    respectively), so plain `torch.stack` is sufficient.
    """
    batch = {}
    for key in PAPER_TENSOR_KEYS:
        batch[key] = torch.stack([item[key] for item in dataset_items])
    batch["question_id"] = [item["question_id"] for item in dataset_items]
    return batch


def gqa_collate_fn(dataset_items: list) -> dict:
    """
    Collate a list of GQA dataset items into a batch.

    All tensor fields must have the same shape across items in the list
    (GQADataset and GQADemoDataset ensure this via padding/trimming).

    Args:
        dataset_items (list[dict]): items from GQADataset.__getitem__.
    Returns:
        batch (dict): batched tensors and metadata.
    """
    tensor_keys = [
        "visual_features",
        "question_input_ids",
        "question_attention_mask",
        "graph_node_features",
        "graph_adj",
        "graph_edge_types",
        "graph_node_types",
        "labels",
    ]

    batch = {}
    for key in tensor_keys:
        batch[key] = torch.stack([item[key] for item in dataset_items])

    batch["question_id"] = [item["question_id"] for item in dataset_items]

    return batch
