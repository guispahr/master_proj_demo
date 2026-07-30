from utils.registry import Registry
import torch
import torch.nn as nn

LOSSES = Registry("losses")

from .lovasz import LovaszLoss
from .aux_geo_loss import AuxGeoLoss
from .distillation_loss import DistillationPassthroughLoss

# from .infonce import InfoNCE
# from .soft_iou import SoftIoU
# from .sincere import SINCERE
# from .geometric_losses import GeometricConsistencyLoss
# from .voxel_losses import VoxelGeometricLoss
# from .aux_2d_losses import Aux2DSegLoss, Aux2DDepthLoss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss that accepts weight as a plain list (from YAML config)
    and moves it to the correct device automatically on the first forward pass."""

    def __init__(self, weight=None, **kwargs):
        if isinstance(weight, (list, tuple)):
            weight = torch.tensor(weight, dtype=torch.float32)
        super().__init__(weight=weight, **kwargs)

    def forward(self, input, target):
        if self.weight is not None and self.weight.device != input.device:
            self.weight = self.weight.to(input.device)
        return super().forward(input, target)


LOSSES.register_module(module=CrossEntropyLoss)
LOSSES.register_module(module=LovaszLoss)
LOSSES.register_module(module=AuxGeoLoss)
LOSSES.register_module(module=DistillationPassthroughLoss)


# LOSSES.register_module(module=SoftIoU)

# LOSSES.register_module(module=InfoNCE)
# LOSSES.register_module(module=SINCERE)

# LOSSES.register_module(module=Aux2DSegLoss)
# LOSSES.register_module(module=Aux2DDepthLoss)

# LOSSES.register_module(module=GeometricConsistencyLoss)
# LOSSES.register_module(module=VoxelGeometricLoss)



