import sys
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
import torch.nn as nn

from models.utils import Point
from models.builder import MODULES


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    

class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    scf = input.sparse_conv_feat
                    # spconv cuDNN tuning fails for bfloat16 in eval mode.
                    # Cast to float32 for the conv and restore dtype after.
                    orig_dtype = scf.features.dtype
                    need_fp32 = (not module.training) and (orig_dtype != torch.float32)
                    if need_fp32:
                        scf = scf.replace_feature(scf.features.float())
                    scf = module(scf)
                    if need_fp32:
                        scf = scf.replace_feature(scf.features.to(orig_dtype))
                    input.sparse_conv_feat = scf
                    input.feat = scf.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input

@MODULES.register_module()
class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point
    

# Original implementation
# class PointSequential(PointModule):
#     r"""A sequential container.
#     Modules will be added to it in the order they are passed in the constructor.
#     Alternatively, an ordered dict of modules can also be passed in.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         if len(args) == 1 and isinstance(args[0], OrderedDict):
#             for key, module in args[0].items():
#                 self.add_module(key, module)
#         else:
#             for idx, module in enumerate(args):
#                 self.add_module(str(idx), module)
#         for name, module in kwargs.items():
#             if sys.version_info < (3, 6):
#                 raise ValueError("kwargs only supported in py36+")
#             if name in self._modules:
#                 raise ValueError("name exists.")
#             self.add_module(name, module)

#     def __getitem__(self, idx):
#         if not (-len(self) <= idx < len(self)):
#             raise IndexError("index {} is out of range".format(idx))
#         if idx < 0:
#             idx += len(self)
#         it = iter(self._modules.values())
#         for i in range(idx):
#             next(it)
#         return next(it)

#     def __len__(self):
#         return len(self._modules)

#     def add(self, module, name=None):
#         if name is None:
#             name = str(len(self._modules))
#             if name in self._modules:
#                 raise KeyError("name exists")
#         self.add_module(name, module)

#     def forward(self, input):
#         for k, module in self._modules.items():
#             # Point module
#             if isinstance(module, PointModule):
#                 input = module(input)
#             # Spconv module
#             elif spconv.modules.is_spconv_module(module):
#                 if isinstance(input, Point):
#                     input.sparse_conv_feat = module(input.sparse_conv_feat)
#                     input.feat = input.sparse_conv_feat.features
#                 else:
#                     input = module(input)
#             # PyTorch module
#             else:
#                 if isinstance(input, Point):
#                     input.feat = module(input.feat)
#                     if "sparse_conv_feat" in input.keys():
#                         input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
#                             input.feat
#                         )
#                 elif isinstance(input, spconv.SparseConvTensor):
#                     if input.indices.shape[0] != 0:
#                         input = input.replace_feature(module(input.features))
#                 else:
#                     input = module(input)
#         return input