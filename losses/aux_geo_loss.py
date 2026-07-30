import torch
import torch.nn as nn
import torch.nn.functional as F

from .lovasz import LovaszLoss


class AuxGeoLoss(nn.Module):
    """Composite loss (CE + optional Lovasz + optional KL alignment) on the
    auxiliary geometric decoder output (``point["geo_logits"]``).

    The geo decoder has no image access, so the CE/Lovasz terms force the
    shared encoder to retain geometrically useful features even when image
    fusion is available. Possible KL divergence term between the geo-head 
    and main-head probability distributions, pushing the geometric branch to 
    mimic the fused branch's output while staying image-free — a 
    self-distillation regularizer that defends against image over-reliance.

    Defaults reproduce the original CE-only behavior (lovasz/alignment off).

    NEEDS_POINT_DICT=True: LossManager passes the full Point dict without a
    separate targets argument (segment labels are read from point["segment"]).
    All components consume ``geo_logits``, ``logits`` (when alignment is on),
    and ``point["segment"]``.

    Returns a zero scalar gracefully when geo_logits is absent
    """

    NEEDS_POINT_DICT = True

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        ce_weight: float = 1.0,
        lovasz_weight: float = 0.0,
        lovasz_mode: str = "multiclass",
        lovasz_per_image: bool = False,
        alignment_weight: float = 0.0,
        alignment_temperature: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.lovasz_weight = float(lovasz_weight)
        self.alignment_weight = float(alignment_weight)
        self.alignment_temperature = float(alignment_temperature)
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        self.lovasz = (
            LovaszLoss(
                mode=lovasz_mode,
                ignore_index=ignore_index,
                per_image=lovasz_per_image,
                loss_weight=1.0,
            )
            if self.lovasz_weight > 0
            else None
        )

    def forward(self, point_dict, targets=None):
        if not isinstance(point_dict, dict):
            return point_dict.new_zeros(())
        if "geo_logits" not in point_dict.keys():
            return point_dict["feat"].new_zeros(())
        geo_logits = point_dict["geo_logits"]
        segment = (
            point_dict["segment"] if "segment" in point_dict.keys() else targets
        )
        segment = segment.long()

        ce_loss = (
            self.ce(geo_logits, segment)
            if self.ce_weight > 0
            else geo_logits.new_zeros(())
        )
        lov_loss = (
            self.lovasz(geo_logits, segment)
            if self.lovasz is not None
            else geo_logits.new_zeros(())
        )

        # KL alignment from geo branch toward main branch.
        align_loss = geo_logits.new_zeros(())
        if self.alignment_weight > 0 and "logits" in point_dict.keys():
            main_logits = point_dict["logits"]
            T = self.alignment_temperature
            valid = segment != self.ignore_index
            if valid.any():
                gl = geo_logits[valid] / T
                ml = main_logits[valid].detach() / T
                kl = F.kl_div(
                    F.log_softmax(gl, dim=-1),
                    F.softmax(ml, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                align_loss = kl

        return (
            self.ce_weight * ce_loss
            + self.lovasz_weight * lov_loss
            + self.alignment_weight * align_loss
        )
