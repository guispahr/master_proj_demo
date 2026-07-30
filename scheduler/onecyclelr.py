from torch.optim import lr_scheduler


class OneCycleLRScheduler:
    """Thin wrapper around torch.optim.lr_scheduler.OneCycleLR.

    Matches the interface expected by the trainer:
    - step() with no arguments (called once per optimizer step)
    - last_step property (used for logging)
    - state_dict() / load_state_dict() for checkpoint resume

    Per-group max_lr is derived from the optimizer's current LRs so that
    param_dicts groups (e.g. a lower-LR encoder group) are scaled
    proportionally, exactly like LinearWarmupCosineScheduler.
    """

    def __init__(
        self,
        optimizer,
        *,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_step: int = -1,
    ):
        base_peak = optimizer.param_groups[0]["lr"]
        per_group_max = [max_lr * pg["lr"] / base_peak for pg in optimizer.param_groups]

        # PyTorch only accepts "cos" or "linear"; normalise the common alias
        if anneal_strategy == "cosine":
            anneal_strategy = "cos"

        self._scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=per_group_max,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_step,
        )

    @property
    def last_step(self):
        return self._scheduler.last_epoch

    def step(self, step=None):
        self._scheduler.step()

    def state_dict(self):
        return {"last_step": self.last_step, "inner": self._scheduler.state_dict()}

    def load_state_dict(self, sd):
        if "inner" in sd:
            self._scheduler.load_state_dict(sd["inner"])
        # if loading from a LinearWarmupCosine checkpoint there is no "inner" —
        # silently ignore so training continues from step 0 of the new scheduler
