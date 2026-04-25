"""
Paper-exact optimizer parameter-group construction for VQA-GNN (arXiv:2205.11501, §5.1).

The paper assigns two distinct learning rates — one to the LM encoder, one to
"VQA-GNN" (every other parameter) — in both VCR and GQA blocks of §5.1:

    VCR (verbatim):
      "finetune it [the pretrained RoBERTa Large] with the multimodal GNN for
      50 epoch by using learning rates 1e-5 and 1e-4 respectively."

    GQA (verbatim):
      "finetune the node q with VQA-GNN for 50 epoch by using learning rates
      2e-5 and 2e-4 respectively."

This helper walks a model's named parameters and partitions them into exactly
two AdamW-compatible parameter groups:

    Group "lm": every parameter whose name starts with `<lm_prefix>.`.
    Group "gnn": every other trainable parameter.

The caller passes the two LRs explicitly (the utility does not hard-code VCR
vs. GQA), so the same function is used by both task-specific configs and
cannot drift.

Rationale for prefix-matching (instead of `isinstance` on a transformers
class):
    * the paper treats the text encoder as a single object with its own LR,
      and the repo's `HFQuestionEncoder.encoder` is always the wrapped HF
      model regardless of text encoder choice;
    * prefix-matching stays stable under HF checkpoint / class changes.

Default prefix `question_encoder.encoder` matches both
`PaperVCRModel.question_encoder.encoder.*` and
`PaperGQAModel.question_encoder.encoder.*`.

Tested by `tests/test_paper_optim.py::test_two_param_groups_split`.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def paper_param_groups(
    model: nn.Module,
    *,
    lm_lr: float,
    other_lr: float,
    lm_prefix: str = "question_encoder.encoder",
    skip_frozen: bool = True,
) -> list[dict]:
    """
    Build the two-LR parameter groups mandated by paper §5.1.

    Args:
        model: the model whose parameters are to be optimized.
        lm_lr: learning rate for the LM encoder group.
        other_lr: learning rate for every other parameter.
        lm_prefix: dotted-name prefix identifying the LM sub-module. The
            default matches both `PaperVCRModel` and `PaperGQAModel` (and the
            legacy `VCRVQAGNNModel`, `GQAVQAGNNModel`). A parameter belongs
            to the LM group iff its `named_parameters()` name starts with
            ``f"{lm_prefix}."`` — the trailing dot is added automatically.
        skip_frozen: if True (default), parameters with `requires_grad=False`
            are omitted entirely. This avoids registering frozen LM weights
            with the optimizer.

    Returns:
        A list `[lm_group, other_group]` suitable for
        `torch.optim.AdamW(groups, ...)`. Each entry is a dict with keys
        `params`, `lr`, and a diagnostic `name`. Empty groups are pruned.

    Raises:
        ValueError: if both groups end up empty (i.e. the model has no
            trainable parameters), which is almost certainly a misuse.
    """
    if lm_lr <= 0:
        raise ValueError(f"lm_lr must be > 0, got {lm_lr}")
    if other_lr <= 0:
        raise ValueError(f"other_lr must be > 0, got {other_lr}")

    prefix = lm_prefix if lm_prefix.endswith(".") else f"{lm_prefix}."

    lm_params: list = []
    other_params: list = []

    for name, param in model.named_parameters():
        if skip_frozen and not param.requires_grad:
            continue
        if name.startswith(prefix):
            lm_params.append(param)
        else:
            other_params.append(param)

    groups: list[dict] = []
    if lm_params:
        groups.append({"params": lm_params, "lr": lm_lr, "name": "lm"})
    if other_params:
        groups.append({"params": other_params, "lr": other_lr, "name": "gnn"})

    if not groups:
        raise ValueError(
            "paper_param_groups produced zero groups — the model has no "
            "trainable parameters under the given `lm_prefix` / freeze rules."
        )

    return groups


def count_params_in_group(group: dict) -> int:
    """Small helper for diagnostics / tests: count trainable elements in a group."""
    return sum(p.numel() for p in _as_iterable(group["params"]))


def _as_iterable(params) -> Iterable:
    # AdamW accepts either a list of Tensors or a generator; normalise.
    return list(params)
