"""
GQA (Generalized Visual Question Answering) loss for VQA-GNN (arXiv:2205.11501).

GQA has a single ground-truth answer per question (unlike VQA v2 which has
10 annotators and uses soft labels). The standard loss is plain cross-entropy
with the hard label.

Note on GQA vs. VQA v2 loss:
    VQA v2 uses soft binary CE because each question has up to 10 human answers.
    GQA has exactly one verified answer per question.
    Using hard CE for GQA is therefore the correct choice and not an approximation.

Batch keys consumed:
    logits (Tensor[B, num_answers]): raw logits from model.
    labels (Tensor[B]): hard answer index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GQALoss(nn.Module):
    """
    Hard cross-entropy loss for GQA open-ended QA.

    GQA uses a fixed answer vocabulary (default 1842 answers for the paper-aligned setup).
    Each question has exactly one correct answer.

    Batch keys consumed:
        logits (Tensor[B, num_answers]): raw answer logits from model.
        labels (Tensor[B]): GT answer index.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **batch,
    ) -> dict:
        """
        Compute GQA cross-entropy loss.

        Args:
            logits (Tensor): model output [B, num_answers].
            labels (Tensor): GT labels [B].
        Returns:
            losses (dict): dict with key 'loss'.
        """
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss}
