import torch.nn as nn

from models.utils.structure import Point

from .builder import MODELS, build_model

@MODELS.register_module("Base-Segmentor")
class BaseSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat
        logits = self.seg_head(feat)
        return logits


@MODELS.register_module("Base-Segmentor-V2")
class PointSegmentor(nn.Module):
    """Variant of Base-Segmentor that returns the full Point dict with 'logits' inserted.

    The forward pass returns the Point object (a dict-like structure) with all
    original fields (coord, feat, offset, ...) plus a 'logits' key containing the
    segmentation head output (N, num_classes).

    Required when using GeometricConsistencyLoss, which needs 'coord' in addition
    to 'logits'. The LossManager automatically routes the full dict only to losses
    that declare NEEDS_POINT_DICT=True; all other losses receive just the logits.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        point["logits"] = self.seg_head(point.feat)
        return point


@MODELS.register_module("Point-Segmentor-GeoAux")
class PointSegmentorGeoAux(nn.Module):
    """Segmentor with a main head and an auxiliary geometric decoder head.

    The backbone (e.g. LitePT-Ditr-v2-E with geo_dec=True) returns a Point dict
    that carries both feat (main image-fused features) and geo_feat (pure
    3D geometric features from the auxiliary decoder).  This wrapper applies a
    separate linear head to each and stores the results as logits and
    geo_logits, which downstream losses (CrossEntropyLoss on logits,
    AuxGeoLoss on geo_logits) can consume independently.

    Args:
        num_classes:            number of output classes.
        backbone_out_channels:  channel width of backbone feat (main decoder output).
        geo_out_channels:       channel width of geo_feat (backbone.geo_out_channels).
        backbone:               config dict for the backbone model.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_out_channels: int,
        geo_out_channels: int,
        backbone=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.geo_seg_head = (
            nn.Linear(geo_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        point["logits"] = self.seg_head(point.feat)
        if "geo_feat" in point.keys():
            point["geo_logits"] = self.geo_seg_head(point["geo_feat"])
        return point