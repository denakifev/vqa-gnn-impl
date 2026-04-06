"""
VQA loss for VQA-GNN (arXiv:2205.11501).

Uses soft binary cross-entropy with VQA v2 soft labels, which is
the standard loss for VQA v2 open-ended evaluation.

VQA v2 soft labels: score = min(count / 3, 1.0) for each answer.
Binary CE over the full answer vocabulary treats each answer as an
independent binary prediction, weighted by the soft score.

Reference:
    Anderson et al., Bottom-Up and Top-Down Attention for Image Captioning
    and Visual Question Answering, CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQALoss(nn.Module):
    """
    Soft binary cross-entropy loss for VQA v2.

    Computes per-class binary CE between sigmoid(logits) and answer_scores,
    then averages over all answers and the batch.

    This matches the standard VQA v2 training loss used in most VQA baselines
    (e.g., Anderson et al. 2018, UpDn). Some works (including VQA-GNN) may
    use regular cross-entropy with the hard label; both are supported via
    the `use_soft_labels` flag.

    Batch keys consumed:
        logits (Tensor[B, num_answers]): raw answer logits from model.
        answer_scores (Tensor[B, num_answers]): soft labels in [0, 1].
        labels (Tensor[B]): hard answer index (used only if use_soft_labels=False).
    """

    def __init__(self, use_soft_labels: bool = True):
        """
        Args:
            use_soft_labels (bool): if True, use soft binary CE (VQA v2 standard).
                If False, use hard cross-entropy with the argmax label.
        """
        super().__init__()
        self.use_soft_labels = use_soft_labels

    def forward(
        self,
        logits: torch.Tensor,
        answer_scores: torch.Tensor,
        labels: torch.Tensor,
        **batch,
    ) -> dict:
        """
        Compute the VQA loss.

        Args:
            logits (Tensor): model output [B, num_answers].
            answer_scores (Tensor): soft labels [B, num_answers].
            labels (Tensor): hard labels [B] (used when use_soft_labels=False).
        Returns:
            losses (dict): dict with key 'loss'.
        """
        if self.use_soft_labels:
            # Soft binary CE: standard VQA v2 training objective
            loss = F.binary_cross_entropy_with_logits(
                logits, answer_scores, reduction="mean"
            )
        else:
            # Hard CE: simpler, less aligned with VQA evaluation
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss}
