"""
Multi-scale encoder distillation wrapper.

Trains a student backbone (e.g. LitePT-Ditr-v2) on the usual segmentation task
while pulling its per-stage encoder features toward a frozen Utonia teacher. Both
backbones pool with the same GridPooling, so for a given input they produce
features at the same points in the same order; only the channel widths differ,
which the per-stage projection heads bridge.

All the distillation settings live here and the student itself is untouched. The
wrapper adds a distill_loss to the returned Point dict, which the standard
LossManager picks up via DistillationPassthroughLoss.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import MODELS, build_model
from models.utils.structure import Point
from models.litept_ditr_v2.litept_v1 import MLP
from models.utonia.point_transformer_utonia import (
    PointTransformerV3 as UtoniaModel,
    load as load_utonia,
)


@MODELS.register_module("Distiller-Segmentor")
class DistillerSegmentor(nn.Module):
    """Segmentor that distills a frozen Utonia teacher into the student's encoder.

    At build time it creates the student, loads and freezes a pretrained Utonia
    checkpoint, and hooks the matching encoder stages of both so their features are
    captured on every forward. In forward it projects the student features into the
    teacher's channel space and adds a per-stage alignment loss (cosine, smooth_l1,
    or hybrid). The seg head and losses share the student's backward graph; the
    teacher runs under no_grad and contributes no gradients.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_out_channels: int,
        student: dict,
        teacher_ckpt: str = "utonia",
        teacher_repo_id: str = "Pointcept/utonia",
        teacher_config_overrides: dict | None = None,
        teacher_feat_keys: tuple[str, ...] = ("coord", "color", "normal"),
        distill_stages: tuple[int, ...] = (0, 1, 2, 3, 4),
        per_stage_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
        mode: Literal["cosine", "smooth_l1", "hybrid"] = "cosine",
        proj_hidden_ratio: float = 2.0,
        proj_layer_norm: bool = False,
        proj_dropout: float = 0.0,
        detach_teacher: bool = True,
    ):
        """Build the student, load and freeze the teacher, and set up the projection
        heads, seg head, and the hooks that capture each encoder stage's features.

        The teacher is the published Utonia backbone (pure 3D, no image branch); only
        enc_mode is forced on, the rest comes from its checkpoint. Per-stage weights
        live in a registered buffer so they follow .to(device). The projection heads
        are two-layer MLPs sized to the teacher's widths, with dropout and a final
        LayerNorm off by default.
        """
        super().__init__()
        assert len(distill_stages) == len(per_stage_weights), (
            f"distill_stages ({len(distill_stages)}) and per_stage_weights "
            f"({len(per_stage_weights)}) must have the same length"
        )
        assert mode in ("cosine", "smooth_l1", "hybrid"), f"unknown mode={mode!r}"

        self.distill_stages = tuple(distill_stages)
        self.mode = mode
        self.detach_teacher = detach_teacher
        self.teacher_feat_keys = tuple(teacher_feat_keys)

        self.register_buffer(
            "per_stage_weights",
            torch.tensor(per_stage_weights, dtype=torch.float32),
            persistent=False,
        )

        # Student backbone.
        self.student = build_model(student)
        student_enc_channels = self._read_enc_channels(self.student, student)

        # Teacher backbone (frozen, encoder-only).
        teacher_overrides = dict(teacher_config_overrides or {})
        teacher_overrides.setdefault("enc_mode", True)

        ckpt = load_utonia(teacher_ckpt, repo_id=teacher_repo_id, ckpt_only=True)
        teacher_cfg = dict(ckpt["config"])
        teacher_cfg.update(teacher_overrides)
        self.teacher = UtoniaModel(**teacher_cfg)

        backbone_state = {
            k: v for k, v in ckpt["state_dict"].items()
            if k.startswith(("embedding.", "enc."))
        }
        missing, unexpected = self.teacher.load_state_dict(backbone_state, strict=False)
        print(
            f"[Distiller-Segmentor] Loaded {len(backbone_state)} teacher params from "
            f"'{teacher_ckpt}' (missing={len(missing)}, unexpected={len(unexpected)})"
        )

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        teacher_enc_channels = teacher_cfg["enc_channels"]
        for s in self.distill_stages:
            assert 0 <= s < len(student_enc_channels), (
                f"distill_stage {s} out of range for student "
                f"(num_stages={len(student_enc_channels)})"
            )
            assert 0 <= s < len(teacher_enc_channels), (
                f"distill_stage {s} out of range for teacher "
                f"(num_stages={len(teacher_enc_channels)})"
            )

        # Per-stage projection heads.
        self.distill_proj_heads = nn.ModuleList([
            self._make_proj_head(
                in_c=student_enc_channels[s],
                hidden_c=int(teacher_enc_channels[s] * proj_hidden_ratio),
                out_c=teacher_enc_channels[s],
                dropout=proj_dropout,
                layer_norm=proj_layer_norm,
            )
            for s in self.distill_stages
        ])

        # Segmentation head on the student's final decoder output.
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # Forward hooks that capture each encoder stage's output feature tensor.
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._warned_missing_keys: set[str] = set()

        for s in self.distill_stages:
            student_stage = getattr(self.student.enc, f"enc{s}")
            student_stage.register_forward_hook(self._make_hook(self._student_feats, s))

            teacher_stage = getattr(self.teacher.enc, f"enc{s}")
            teacher_stage.register_forward_hook(self._make_hook(self._teacher_feats, s))

    @staticmethod
    def _make_proj_head(in_c: int, hidden_c: int, out_c: int,
                        dropout: float, layer_norm: bool) -> nn.Module:
        """Build a per-stage projection head: an MLP plus an optional LayerNorm."""
        mlp = MLP(
            in_channels=in_c,
            hidden_channels=hidden_c,
            out_channels=out_c,
            act_layer=nn.GELU,
            drop=dropout,
        )
        if layer_norm:
            return nn.Sequential(mlp, nn.LayerNorm(out_c))
        return mlp

    @staticmethod
    def _read_enc_channels(_model, cfg: dict) -> list[int]:
        """Return the encoder channel list from the backbone's YAML config."""
        if "enc_channels" not in cfg:
            raise KeyError(
                "DistillerSegmentor requires enc_channels in the student YAML config "
                "to size the per-stage projection heads."
            )
        return list(cfg["enc_channels"])

    @staticmethod
    def _make_hook(store: dict, idx: int):
        """Return a forward hook that stashes a stage's output feature tensor.

        Grabs the post-stage feat directly, so later in-place feat rewrites (e.g. by
        GridUnpooling) don't clobber the captured tensor.
        """
        def hook(module, _input, output):
            store[idx] = output.feat
        return hook

    def _build_teacher_input(self, input_dict: dict) -> dict:
        """Return a teacher-ready copy of the data dict.

        The teacher takes different input channels than the student (e.g.
        coord+color+normal vs coord+color), so feat is rebuilt from teacher_feat_keys,
        filling any missing key (NuScenes has no color/normal) with zeros. Serialization
        and sparse-conv fields are dropped from the copy so the teacher's in-place
        edits, run under no_grad, never leak back into the student's dict.
        """
        cloned = {k: v for k, v in input_dict.items() if not k.startswith("serialized_")}
        cloned.pop("sparse_conv_feat", None)

        parts = []
        for k in self.teacher_feat_keys:
            if k not in cloned:
                if k not in self._warned_missing_keys:
                    print(f"[Distiller-Segmentor] Warning: teacher_feat_keys contains '{k}' "
                          f"which is absent from the input - substituting zeros.")
                    self._warned_missing_keys.add(k)
                v = torch.zeros_like(cloned["coord"])
            else:
                v = cloned[k]
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            parts.append(v.float())
        cloned["feat"] = torch.cat(parts, dim=-1)
        return cloned

    def _per_stage_loss(self, s_proj: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Compute the per-stage feature-alignment loss for the configured mode."""
        s = s_proj.float()
        t = t_feat.float()
        if self.mode == "cosine":
            return 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        if self.mode == "smooth_l1":
            return F.smooth_l1_loss(s, t)
        # hybrid: sum of both
        return (1.0 - F.cosine_similarity(s, t, dim=-1).mean()) + F.smooth_l1_loss(s, t)

    @staticmethod
    def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Linear CKA between two feature sets, in [0, 1] (higher = better aligned).

        Scale- and dimension-invariant, so it compares student and teacher features
        even at different channel widths, without going through the trainable
        projection head. Linear-kernel CKA (Kornblith et al. 2019) on mean-centered
        inputs, in fp32 to avoid bf16 drift; the small C x C matrices make it cheap.
        """
        x = x.float()
        y = y.float()
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        yx = y.t() @ x                                   # (Cy, Cx)
        xx_norm = torch.linalg.norm(x.t() @ x)
        yy_norm = torch.linalg.norm(y.t() @ y)
        return (yx.pow(2).sum()) / (xx_norm * yy_norm + 1e-8)

    def forward(self, input_dict):
        """Run teacher and student and return the student's Point dict, plus losses.

        Resets the hook buffers, runs the teacher under no_grad on its own copy of the
        input, then the student (which may edit input_dict in place). For each distilled
        stage it projects the student features, adds the weighted alignment loss, and
        records a CKA diagnostic on the raw (pre-projection) features. Finally it attaches
        the seg-head logits, the total distill_loss, and the per-stage loss/CKA tensors.
        """
        self._student_feats.clear()
        self._teacher_feats.clear()

        teacher_input = self._build_teacher_input(input_dict)
        with torch.no_grad():
            self.teacher(teacher_input)
        teacher_feats = {s: self._teacher_feats[s] for s in self.distill_stages}

        point = self.student(input_dict)
        student_feats = {s: self._student_feats[s] for s in self.distill_stages}

        per_stage_losses = []
        per_stage_cka = []
        for k, s in enumerate(self.distill_stages):
            s_feat = student_feats[s]
            t_feat = teacher_feats[s]
            if self.detach_teacher:
                t_feat = t_feat.detach()
            if s_feat.shape[0] != t_feat.shape[0]:
                raise RuntimeError(
                    f"stage {s}: student feat has {s_feat.shape[0]} points, "
                    f"teacher feat has {t_feat.shape[0]} - check that both backbones "
                    f"share the same GridPooling logic and that the data augmentation "
                    f"pipeline is deterministic across the student/teacher calls."
                )
            s_proj = self.distill_proj_heads[k](s_feat)
            per_stage_losses.append(self._per_stage_loss(s_proj, t_feat))
            with torch.no_grad():
                per_stage_cka.append(self._linear_cka(s_feat, t_feat))

        weights = self.per_stage_weights[: len(per_stage_losses)].to(
            per_stage_losses[0].device, per_stage_losses[0].dtype
        )
        distill_loss = sum(w * l for w, l in zip(weights, per_stage_losses))

        if not isinstance(point, Point):
            point = Point(point)
        point["logits"] = self.seg_head(point.feat)
        point["distill_loss"] = distill_loss
        point["distill_per_stage"] = torch.stack(per_stage_losses).detach()
        point["distill_cka_per_stage"] = torch.stack(per_stage_cka).detach()
        return point

    def train(self, mode: bool = True):
        """Override .train() so the teacher always stays in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self
