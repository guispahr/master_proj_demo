"""Passthrough for the distillation loss computed inside DistillerSegmentor.

The DistillerSegmentor wrapper computes the full multi-scale distillation loss
during its forward pass (it owns the per-stage projection heads, weights, and
mode), then stores the scalar under `point["distill_loss"]`. This passthrough
loss simply surfaces that scalar to the LossManager so it flows through the
standard log/aggregate path alongside CrossEntropy, Lovasz, etc.

Returns a zero scalar gracefully when `distill_loss` is absent.
"""

import torch.nn as nn


class DistillationPassthroughLoss(nn.Module):
    NEEDS_POINT_DICT = True

    def __init__(self, key: str = "distill_loss"):
        super().__init__()
        self.key = key

    def forward(self, point_dict, targets=None):
        if not isinstance(point_dict, dict):
            return point_dict.new_zeros(())
        if self.key not in point_dict.keys():
            return point_dict["feat"].new_zeros(())
        return point_dict[self.key]
