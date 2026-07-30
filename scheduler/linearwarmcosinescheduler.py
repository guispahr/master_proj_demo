# DINOv3 based scheduler
# Linear lr and cosine annealing lr

import numpy as np
from typing import Optional


class LinearWarmupCosineScheduler:
    """
    Step-based learning rate scheduler with:
      - linear warmup
      - cosine decay
      - optional constant tail

    Usage:
        scheduler = LinearWarmupCosineScheduler(...)
        for step in range(total_steps):
            scheduler.step(step)
            optimizer.step()
    """

    def __init__(
        self,
        optimizer,
        *,
        start_lr: float,
        peak_lr: float,
        end_lr: float,
        warmup_steps: int,
        total_steps: int,
        cosine_steps: Optional[int] = None,
        last_step: int = -1,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.last_step = last_step

        # Store each group's peak LR so step() can scale proportionally.
        for pg in optimizer.param_groups:
            pg.setdefault("_peak_lr", pg["lr"])

        self.schedule = self._build_schedule(
            start_lr=start_lr,
            peak_lr=peak_lr,
            end_lr=end_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            cosine_steps=cosine_steps,
        )

    @staticmethod
    def _build_schedule(
        *,
        start_lr: float,
        peak_lr: float,
        end_lr: float,
        warmup_steps: int,
        total_steps: int,
        cosine_steps: Optional[int],
    ) -> np.ndarray:
        assert warmup_steps >= 0
        assert total_steps > warmup_steps

        # --- linear warmup ---
        warmup = np.linspace(
            start_lr,
            peak_lr,
            warmup_steps,
            endpoint=False,
            dtype=np.float64,
        )

        # --- cosine decay ---
        if cosine_steps is None:
            cosine_steps = total_steps - warmup_steps

        t = np.linspace(0, np.pi, cosine_steps, dtype=np.float64)
        cosine = (np.cos(t) + 1.0) / 2.0
        cosine = (peak_lr - end_lr) * cosine + end_lr

        # --- optional constant tail ---
        remaining = total_steps - warmup_steps - cosine_steps
        assert remaining >= 0
        tail = np.full(remaining, end_lr, dtype=np.float64)

        return np.concatenate([warmup, cosine, tail])

    def step(self, step: Optional[int] = None):
        """Update optimizer learning rates for all param groups."""
        if step is None:
            step = self.last_step + 1

        self.last_step = step
        base_lr = float(self.schedule[min(step, self.total_steps - 1)])
        base_peak = self.optimizer.param_groups[0]["_peak_lr"]

        for pg in self.optimizer.param_groups:
            pg["lr"] = base_lr * pg["_peak_lr"] / base_peak

        return base_lr

    def state_dict(self):
        return {
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict):
        self.last_step = state_dict["last_step"]
