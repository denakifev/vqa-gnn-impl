"""
Collate function for VQA-GNN batches.

All tensor fields are expected to have fixed sizes (padded/trimmed by
VQADataset / VQADemoDataset), so we simply stack them.
Non-tensor fields (question_id) are collected into a Python list.
"""

import torch


def vqa_collate_fn(dataset_items: list) -> dict:
    """
    Collate a list of VQA dataset items into a batch.

    All tensor fields must have the same shape across items in the list
    (VQADataset and VQADemoDataset ensure this via padding/trimming).

    Args:
        dataset_items (list[dict]): items from VQADataset.__getitem__.
    Returns:
        batch (dict): batched tensors and metadata.
    """
    tensor_keys = [
        "visual_features",
        "question_input_ids",
        "question_attention_mask",
        "graph_node_features",
        "graph_adj",
        "graph_node_types",
        "answer_scores",
        "labels",
    ]

    batch = {}
    for key in tensor_keys:
        batch[key] = torch.stack([item[key] for item in dataset_items])

    # Non-tensor fields
    batch["question_id"] = [item["question_id"] for item in dataset_items]

    return batch
