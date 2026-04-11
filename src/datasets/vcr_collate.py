"""
Collate function for VCR batches (VQA-GNN, arXiv:2205.11501).

VCR is a 4-way multiple-choice task. The model is evaluated on each
[image, QA-context, KG] triple independently, so a batch of B questions
is expanded to B×4 model inputs.

Expansion strategy:
    For each question (index b in 0..B-1):
        - Candidate 0: (visual_b, qa_input_ids_b[0], graph_b, ...)
        - Candidate 1: (visual_b, qa_input_ids_b[1], graph_b, ...)
        - Candidate 2: (visual_b, qa_input_ids_b[2], graph_b, ...)
        - Candidate 3: (visual_b, qa_input_ids_b[3], graph_b, ...)

The collate output is a single dict whose tensor keys are ready for the
VQAGNNModel forward pass (which uses 'question_input_ids' /
'question_attention_mask' as the QA-context node input).

After forward, VCRLoss reshapes logits from [B×4, 1] → [B, 4] to apply
cross-entropy with the ground-truth candidate index.

task_mode controls which of the two sets of context encodings is used:
    'qa'  → qa_input_ids / qa_attention_mask → question_input_ids / question_attention_mask
    'qar' → qar_input_ids / qar_attention_mask → question_input_ids / question_attention_mask

Non-expanded keys (compact batch size B):
    answer_label        LongTensor[B]   ground-truth index 0–3
    n_candidates        int             always 4 (used by loss/metrics for reshaping)
    annot_ids           list[str]       VCR annotation IDs
"""

import torch

VCR_NUM_CANDIDATES = 4


def make_vcr_collate_fn(task_mode: str = "qa"):
    """
    Factory that returns a VCR collate function for the given task mode.

    Args:
        task_mode (str): 'qa' or 'qar'.
    Returns:
        collate_fn (callable): collate function for use with DataLoader.
    """
    if task_mode not in ("qa", "qar"):
        raise ValueError(f"task_mode must be 'qa' or 'qar', got: {task_mode!r}")

    def vcr_collate_fn(dataset_items: list) -> dict:
        """
        Collate a list of VCR dataset items into a B×4-expanded model batch.

        Args:
            dataset_items (list[dict]): items from VCRDataset.__getitem__.
        Returns:
            batch (dict): model-ready batch with B×4 visual/KG fields and
                          B answer_label tensor.
        """
        B = len(dataset_items)
        NC = VCR_NUM_CANDIDATES

        # --- Shared fields (visual + graph): repeat each sample NC times ---
        shared_tensor_keys = [
            "visual_features",
            "graph_node_features",
            "graph_adj",
            "graph_node_types",
        ]
        # Each item contributes NC copies
        expanded = {k: [] for k in shared_tensor_keys}
        for item in dataset_items:
            for k in shared_tensor_keys:
                for _ in range(NC):
                    expanded[k].append(item[k])
        batch = {k: torch.stack(v) for k, v in expanded.items()}  # [B*NC, ...]

        # --- QA-context fields: interleave candidates ---
        # qa_input_ids shape per item: [4, L]
        ctx_ids_key = "qa_input_ids" if task_mode == "qa" else "qar_input_ids"
        ctx_mask_key = "qa_attention_mask" if task_mode == "qa" else "qar_attention_mask"

        ctx_ids = []
        ctx_masks = []
        for item in dataset_items:
            for c in range(NC):
                ctx_ids.append(item[ctx_ids_key][c])    # [L]
                ctx_masks.append(item[ctx_mask_key][c])  # [L]
        # Rename to the expected model parameter names
        batch["question_input_ids"] = torch.stack(ctx_ids)    # [B*NC, L]
        batch["question_attention_mask"] = torch.stack(ctx_masks)  # [B*NC, L]

        # --- Ground-truth labels (compact, size B) ---
        label_key = "answer_label" if task_mode == "qa" else "rationale_label"
        batch["answer_label"] = torch.stack([item[label_key] for item in dataset_items])  # [B]

        # Metadata
        batch["n_candidates"] = NC
        batch["batch_size"] = B
        batch["annot_ids"] = [item["annot_id"] for item in dataset_items]

        return batch

    return vcr_collate_fn
