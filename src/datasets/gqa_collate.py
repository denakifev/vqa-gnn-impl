"""
Collate function for GQA batches (VQA-GNN, arXiv:2205.11501).

GQA is an open-ended question answering task with a fixed answer vocabulary
(paper target: 1842 answers). Each item has a single ground-truth
answer label (no soft annotations).

This collate is structurally simpler than VCR: no candidate expansion,
just standard tensor stacking.
"""

import torch


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
