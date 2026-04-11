"""
VCR (Visual Commonsense Reasoning) loss for VQA-GNN (arXiv:2205.11501).

VCR is a 4-way multiple-choice task. Each question has exactly 4 candidate
answers (or rationales), exactly one of which is correct.

The training batch is expanded B→B×4 by the VCR collate function
(see src/datasets/vcr_collate.py). The model processes B×4 individual
[image, QA-context, KG] inputs and produces B×4 scalar scores
(model configured with num_answers=1).

This loss:
  1. Reshapes logits from [B×4, 1] to [B, 4].
  2. Applies cross-entropy with the ground-truth candidate index.

Batch keys consumed:
    logits (Tensor[B*4, 1]): raw candidate scores from model.
    answer_label (Tensor[B]): ground-truth candidate index (0–3).
    n_candidates (int): always 4 for VCR (from batch dict).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VCRLoss(nn.Module):
    """
    Cross-entropy loss for VCR 4-way multiple choice.

    Expects the batch to be pre-expanded B→B×4 by vcr_collate_fn.
    The model output is a scalar score per candidate (num_answers=1).

    Batch keys consumed:
        logits (Tensor[B*4, 1]):  candidate scores (num_answers=1 model output).
        answer_label (Tensor[B]): GT candidate index 0–3.
        n_candidates (int):       4 (from collate; used for reshaping).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        answer_label: torch.Tensor,
        n_candidates: int = 4,
        **batch,
    ) -> dict:
        """
        Compute VCR cross-entropy loss.

        Args:
            logits (Tensor): candidate scores [B*n_candidates, 1] or [B*n_candidates].
            answer_label (Tensor): GT label [B], values in 0..n_candidates-1.
            n_candidates (int): number of candidates (always 4 for VCR).
        Returns:
            losses (dict): dict with key 'loss'.
        """
        B = answer_label.size(0)
        # Squeeze trailing dim if present: [B*4, 1] → [B*4]
        scores = logits.view(-1)  # [B*4]
        # Reshape to [B, 4]: candidate scores per question
        scores = scores.view(B, n_candidates)  # [B, 4]
        loss = F.cross_entropy(scores, answer_label)
        return {"loss": loss}
