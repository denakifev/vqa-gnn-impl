"""
Paper-equation learning rate scheduler for VQA-GNN (arXiv:2205.11501, §5.1).

Paper quote (VCR block):
    "We use a linear warmup of the learning rate over the 15-th epoch, with a
    cosine decay thereafter to 0."

Natural literal reading:
    * linear ramp from 0 to the optimizer's configured peak LR over the first
      `warmup_steps` optimization steps (equivalent to 15 epochs when
      warmup_steps = 15 * steps_per_epoch);
    * cosine decay from peak LR to 0 over the remaining
      `total_steps - warmup_steps` steps.

Known underspecifications (see BLOCKED_EXACTNESS_ITEMS.md §C.6, §C.7):
    * the paper says "linear" for warmup — we take that literally;
    * the GQA block of §5.1 does not restate the warmup/cosine schedule,
      so using this class for GQA assumes the VCR schedule transfers directly.
      That assumption is tracked as `not paper-faithful yet`; callers must supply the same
      `warmup_steps` and `total_steps` explicitly for GQA if they use it.

The scheduler operates at *step* granularity so behaviour is independent of the
dataloader length. The standard `Trainer` in this repo calls `step()` once per
optimization step (see `src/trainer/trainer.py`).
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class PaperWarmupCosine(LRScheduler):
    """
    Linear warmup + cosine decay to 0, per paper §5.1.

    Args:
        optimizer: torch optimizer. Each param group's configured `lr` is used
            as that group's peak LR; the schedule is applied uniformly to
            every group (groups keep their *relative* peak LRs intact, e.g.
            the LM group at 1e-5 and the GNN group at 1e-4 both ramp together
            from 0 to their respective peaks).
        warmup_steps: number of steps for the linear 0→peak ramp. Must be >=0.
            A value of 0 disables warmup and starts at the peak LR.
        total_steps: total number of optimizer steps (including warmup).
            After `total_steps`, the LR stays at 0 (step calls past the end
            are safe and simply keep LR at 0).
        last_epoch: as in `torch.optim.lr_scheduler.LRScheduler` — acts as a
            step counter (despite its name). Defaults to -1 (scheduler not yet
            stepped, returns the initial peak at step 0 if warmup_steps == 0,
            or 0 if warmup_steps > 0).

    Example:
        >>> optim = torch.optim.AdamW(params, lr=1e-4)
        >>> sched = PaperWarmupCosine(optim, warmup_steps=1500, total_steps=5000)
        >>> for step in range(5000):
        ...     optim.step()
        ...     sched.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be <= total_steps "
                f"({total_steps}); paper §5.1 places the cosine phase *after* "
                f"warmup, so warmup cannot exceed the total budget."
            )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _scale_at_step(self, step: int) -> float:
        """Multiplier applied to each group's base LR at the given step index.

        step indexing: step 0 is the first call to `scheduler.step()` after
        construction, which maps to `last_epoch == 0` internally.
        """
        if step < 0:
            return 0.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear 0 -> 1 ramp. At step == warmup_steps - 1 the scale is
            # (warmup_steps - 1) / warmup_steps; at step == warmup_steps (first
            # cosine step) the scale reaches 1.0.
            return step / float(self.warmup_steps)
        if step >= self.total_steps:
            return 0.0
        # Cosine decay from 1 at step == warmup_steps to 0 at step == total_steps.
        progress = (step - self.warmup_steps) / float(
            self.total_steps - self.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_lr(self) -> list[float]:
        scale = self._scale_at_step(self.last_epoch)
        return [base_lr * scale for base_lr in self.base_lrs]
