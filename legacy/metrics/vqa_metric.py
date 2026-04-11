"""
VQA accuracy metric for VQA-GNN (arXiv:2205.11501).

Implements the official VQA evaluation metric from:
    Agrawal et al., VQA: Visual Question Answering, ICCV 2015.

Official formula:
    Acc(ans) = min( #humans_that_said_ans / 3, 1 )

In our setting we have soft labels answer_scores ∈ [0, 1] already encoding
this formula. The metric picks the model's top-1 predicted answer and looks
up its soft score:

    vqa_acc = mean_over_batch( answer_scores[argmax(logits)] )

This matches the standard evaluation used in VQA v2 baselines.

Note: the paper may apply slight variants (e.g. per-type accuracy).
For the first-stage baseline we use the overall VQA accuracy.
"""

import torch

from src.metrics.base_metric import BaseMetric


class VQAAccuracy(BaseMetric):
    """
    Official VQA v2 accuracy.

    Measures the fraction of correctly predicted answers weighted by the
    soft VQA label (min(count/3, 1)) of the predicted answer.

    Batch keys consumed:
        logits (Tensor[B, num_answers]): model output logits.
        answer_scores (Tensor[B, num_answers]): soft labels in [0, 1].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        logits: torch.Tensor,
        answer_scores: torch.Tensor,
        **batch,
    ) -> float:
        """
        Compute mean VQA accuracy for the batch.

        Args:
            logits (Tensor): raw model logits [B, num_answers].
            answer_scores (Tensor): soft labels [B, num_answers].
        Returns:
            accuracy (float): mean VQA accuracy in [0, 1].
        """
        pred_idx = logits.argmax(dim=-1)  # [B]
        # Gather the soft score for the predicted answer
        batch_idx = torch.arange(pred_idx.size(0), device=pred_idx.device)
        scores = answer_scores[batch_idx, pred_idx]  # [B]
        return scores.mean().item()
