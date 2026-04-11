"""
GQA accuracy metric for VQA-GNN (arXiv:2205.11501).

GQA uses exact-match accuracy: the predicted answer must match the single
ground-truth answer exactly. There are no soft labels or partial credit.

This differs from the VQA v2 metric (which uses soft labels from multiple
annotators). Using VQA v2 accuracy for GQA would be incorrect.

Reference:
    Hudson & Manning, GQA: A New Dataset for Real-World Visual Reasoning
    and Compositional Question Answering, CVPR 2019.
"""

import torch

from src.metrics.base_metric import BaseMetric


class GQAAccuracy(BaseMetric):
    """
    GQA exact-match accuracy.

    The predicted answer (argmax over logits) must equal the GT answer label.

    Batch keys consumed:
        logits (Tensor[B, num_answers]): model output logits.
        labels (Tensor[B]): GT answer indices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **batch,
    ) -> float:
        """
        Compute mean GQA accuracy for the batch.

        Args:
            logits (Tensor): raw model logits [B, num_answers].
            labels (Tensor): GT answer indices [B].
        Returns:
            accuracy (float): exact-match accuracy in [0, 1].
        """
        pred_idx = logits.argmax(dim=-1)  # [B]
        return (pred_idx == labels).float().mean().item()
