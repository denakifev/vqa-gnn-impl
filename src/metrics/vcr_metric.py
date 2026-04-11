"""
VCR metrics for VQA-GNN (arXiv:2205.11501).

VCR has three standard metrics:

    Q→A accuracy (VCRQAAccuracy)
        Given image + question, choose the correct answer from 4 candidates.
        Accuracy = fraction of questions where the top-scored candidate == answer_label.

    QA→R accuracy (VCRQARAccuracy)
        Given image + question + correct answer, choose the correct rationale.
        Same formula: top-scored candidate == rationale_label.
        Evaluated only on QA→R training runs (separate from Q→A).

    Q→AR joint accuracy (VCRQARJointAccuracy)
        Fraction of questions answered BOTH correctly in Q→A and QA→R.
        This cannot be computed from a single model's output — it requires
        saving predictions from both the Q→A and QA→R models and combining them.
        This metric class supports computing it when both prediction sets are
        available in the batch (e.g., during a combined evaluation pass).
        In standard single-task training, use VCRQAAccuracy or VCRQARAccuracy.

Batch contract (all metrics):
    logits (Tensor[B*4, 1]):   candidate scores from the model (num_answers=1).
    answer_label (Tensor[B]):  GT candidate index 0–3.
    n_candidates (int):        always 4 (from collate batch).
"""

import torch

from src.metrics.base_metric import BaseMetric

VCR_NUM_CANDIDATES = 4


class VCRQAAccuracy(BaseMetric):
    """
    VCR Q→A accuracy: fraction of questions with correct answer prediction.

    Batch keys consumed:
        logits (Tensor[B*4, 1] or [B*4]): candidate scores.
        answer_label (Tensor[B]): GT answer index 0–3.
        n_candidates (int): 4 (from collate).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        logits: torch.Tensor,
        answer_label: torch.Tensor,
        n_candidates: int = VCR_NUM_CANDIDATES,
        **batch,
    ) -> float:
        """
        Args:
            logits (Tensor): [B*n_candidates, 1] or [B*n_candidates].
            answer_label (Tensor): [B].
            n_candidates (int): 4.
        Returns:
            accuracy (float): mean accuracy in [0, 1].
        """
        B = answer_label.size(0)
        scores = logits.view(B, n_candidates)  # [B, 4]
        pred = scores.argmax(dim=-1)           # [B]
        return (pred == answer_label).float().mean().item()


class VCRQARAccuracy(BaseMetric):
    """
    VCR QA→R accuracy: fraction of questions with correct rationale prediction.

    Identical formula to VCRQAAccuracy but semantically applied to the QA→R task.
    This metric is used in QA→R training runs where answer_label contains
    rationale_label (the collate renames rationale_label → answer_label).

    Batch keys consumed:
        logits (Tensor[B*4, 1] or [B*4]): candidate scores.
        answer_label (Tensor[B]): GT rationale index 0–3 (renamed by collate).
        n_candidates (int): 4.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        logits: torch.Tensor,
        answer_label: torch.Tensor,
        n_candidates: int = VCR_NUM_CANDIDATES,
        **batch,
    ) -> float:
        """
        Args:
            logits (Tensor): [B*n_candidates, 1] or [B*n_candidates].
            answer_label (Tensor): [B] (rationale GT, renamed by collate).
            n_candidates (int): 4.
        Returns:
            accuracy (float): mean accuracy in [0, 1].
        """
        B = answer_label.size(0)
        scores = logits.view(B, n_candidates)
        pred = scores.argmax(dim=-1)
        return (pred == answer_label).float().mean().item()


class VCRQARJointAccuracy(BaseMetric):
    """
    VCR Q→AR joint accuracy.

    Requires BOTH the Q→A and QA→R predictions to be correct for a question
    to count as correctly answered.

    This metric is included for completeness and is used when a combined
    evaluation batch contains both prediction sets. In standard separate-task
    training, use VCRQAAccuracy and VCRQARAccuracy individually.

    Batch keys consumed:
        qa_logits (Tensor[B*4, 1]):    Q→A candidate scores.
        qar_logits (Tensor[B*4, 1]):   QA→R candidate scores.
        answer_label (Tensor[B]):      GT answer index 0–3.
        rationale_label (Tensor[B]):   GT rationale index 0–3.
        n_candidates (int):            4.

    NOTE: This metric is NOT the default metric in single-task training configs.
    It is listed here for documentation and potential combined evaluation use.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        qa_logits: torch.Tensor,
        qar_logits: torch.Tensor,
        answer_label: torch.Tensor,
        rationale_label: torch.Tensor,
        n_candidates: int = VCR_NUM_CANDIDATES,
        **batch,
    ) -> float:
        """
        Args:
            qa_logits (Tensor): Q→A scores [B*n_candidates].
            qar_logits (Tensor): QA→R scores [B*n_candidates].
            answer_label (Tensor): GT answer [B].
            rationale_label (Tensor): GT rationale [B].
            n_candidates (int): 4.
        Returns:
            joint_accuracy (float): fraction correct on both tasks.
        """
        B = answer_label.size(0)
        qa_pred = qa_logits.view(B, n_candidates).argmax(dim=-1)
        qar_pred = qar_logits.view(B, n_candidates).argmax(dim=-1)
        both_correct = (qa_pred == answer_label) & (qar_pred == rationale_label)
        return both_correct.float().mean().item()
