import numpy
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy, MulticlassPrecision, MulticlassConfusionMatrix

eps = 1e-6


class SegmentationMetrics:
    """Point cloud segmentation metrics (IoU + Accuracy + Precision) compatible with MetricManager."""

    def __init__(self, num_classes, ignore_index, device="cpu"):
        self.acc_metric = Accuracy(
            task="multiclass", num_classes=num_classes,
            ignore_index=ignore_index, average=None,
        ).to(device)
        self.iou_metric = MulticlassJaccardIndex(
            num_classes=num_classes, average=None, ignore_index=ignore_index,
        ).to(device)
        self.precision_metric = MulticlassPrecision(
            num_classes=num_classes, average=None, ignore_index=ignore_index,
        ).to(device)
        self.confusion_metric = MulticlassConfusionMatrix(
            num_classes=num_classes, ignore_index=ignore_index,
        ).to(device)

    def reset(self):
        self.acc_metric.reset()
        self.iou_metric.reset()
        self.precision_metric.reset()
        self.confusion_metric.reset()

    def update(self, outputs, batch):
        labels = batch["segment"]
        preds = torch.argmax(outputs, dim=1)
        self.acc_metric.update(preds, labels)
        self.iou_metric.update(preds, labels)
        self.precision_metric.update(preds, labels)
        self.confusion_metric.update(preds, labels)

    def compute(self):
        accuracy = self.acc_metric.compute()
        iou = self.iou_metric.compute()
        precision = self.precision_metric.compute()
        confmat = self.confusion_metric.compute()   # (C, C), rows=target, cols=pred

        # Overall Accuracy (OA / allAcc): correct / total over valid points.
        total = confmat.sum()
        all_acc = (confmat.diag().sum().float() / total.clamp(min=1)).item()

        return {
            "accuracy": accuracy.cpu(),
            "mean_accuracy": accuracy.mean().item(),
            "OA": all_acc,
            "iou": iou.cpu(),
            "mIoU": iou.mean().item(),
            "precision": precision.cpu(),
            "mean_precision": precision.mean().item(),
            "confusion_matrix": confmat.cpu(),
        }


class RangeBinnedSegMetrics:
    """Per-point range-binned mIoU (validation only).

    Bins each point by its horizontal distance ``||coord_xy - origin||`` and keeps
    one ``MulticlassJaccardIndex`` per range bin. This is the key diagnostic for
    judging whether a second, image-derived geometry stream helps where the LiDAR
    is sparse (far range): a gain that does not concentrate in the far bins is a
    kill signal for that stream.

    ``bins=(20, 40)`` → three bins ``[0,20) / [20,40) / [40,inf)`` (metres).
    ``origin`` is an optional ``(x, y)`` sensor origin (defaults to ``(0, 0)``,
    correct for ego-centred LiDAR such as nuScenes).

    Plugs into :class:`MetricManager` exactly like :class:`SegmentationMetrics`.
    It is a no-op during training (gated on ``torch.is_grad_enabled()`` — the
    train loop runs with grad enabled, ``validate()`` is ``@torch.no_grad()``) and
    when the metric ``meta`` dict lacks ``coord`` (e.g. the full-resolution val
    path that forwards only ``segment``). Both ``update`` and ``compute`` apply the
    same gate, so under DDP every rank consistently enters (or skips) the
    torchmetrics collective in ``compute``.
    """

    def __init__(self, num_classes, ignore_index, bins=(20.0, 40.0), origin=None, device="cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edges = [float(b) for b in bins]
        self.origin = origin
        self.device = device
        self.bin_metrics = [
            MulticlassJaccardIndex(
                num_classes=num_classes, average=None, ignore_index=ignore_index,
            ).to(device)
            for _ in range(len(self.edges) + 1)
        ]
        self.bin_names = self._bin_names()

    def _bin_names(self):
        names, prev = [], 0
        for e in self.edges:
            names.append(f"{int(prev)}-{int(e)}m")
            prev = e
        names.append(f"{int(prev)}m+")
        return names

    def reset(self):
        for m in self.bin_metrics:
            m.reset()

    def update(self, outputs, batch):
        if torch.is_grad_enabled():          # training — skip (val-only metric)
            return
        if "coord" not in batch:             # full-res val path forwards only segment
            return
        labels = batch["segment"]
        preds = torch.argmax(outputs, dim=1)
        xy = batch["coord"][:, :2].float()
        if self.origin is not None:
            xy = xy - torch.as_tensor(self.origin, device=xy.device, dtype=xy.dtype)
        rng = torch.linalg.norm(xy, dim=1)
        edges = torch.as_tensor(self.edges, device=rng.device, dtype=rng.dtype)
        # bin id = #edges that rng is >= : rng<e0 -> 0, e0<=rng<e1 -> 1, ...
        bin_idx = (rng[:, None] >= edges[None, :]).sum(dim=1)
        for b, m in enumerate(self.bin_metrics):
            sel = bin_idx == b
            if sel.any():
                m.update(preds[sel], labels[sel])

    def compute(self):
        if torch.is_grad_enabled():
            return {}
        out = {}
        for name, m in zip(self.bin_names, self.bin_metrics):
            iou = m.compute()                # (C,) — collective under DDP
            out[f"mIoU@{name}"] = iou.mean().item()
        return out


class BoundarySegMetrics:
    """mIoU restricted to class-boundary points (validation only).

    A point is a *boundary* point if, within its 26-neighbour cell neighbourhood
    on a ``radius``-metre grid, there is a point carrying a different GT label.
    ``radius=0.05`` ≈ the 5 cm boundary band; it measures exactly the edge/thin-
    structure accuracy that overall mIoU dilutes.

    Boundary detection is exact, vectorised and dependency-free (no torch_cluster):
    cells are integer-encoded ``(scene, x, y, z)`` → int64 keys, the unique cells
    are sorted, neighbour cells are located with ``torch.searchsorted``, and a
    per-cell ``(U, num_classes)`` class-presence table (one ``scatter_reduce``)
    answers "does this neighbour cell contain a different label?". Cost is 27
    searchsorted passes over the unique cells — cheap at val resolution.

    Same val-only gating and ``coord`` requirement as :class:`RangeBinnedSegMetrics`;
    also needs ``segment`` and (for multi-scene batches) ``offset`` so points from
    different scenes are never treated as neighbours.
    """

    def __init__(self, num_classes, ignore_index, radius=0.05, device="cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.radius = float(radius)
        self.device = device
        self.metric = MulticlassJaccardIndex(
            num_classes=num_classes, average=None, ignore_index=ignore_index,
        ).to(device)

    def reset(self):
        self.metric.reset()

    @staticmethod
    def _scene_index(n, offset, device):
        """Map each point to its scene id from a cumulative `offset` vector."""
        if offset is None or offset.numel() <= 1:
            return torch.zeros(n, dtype=torch.long, device=device)
        return torch.bucketize(
            torch.arange(n, device=device), offset.to(device), right=True
        )

    def _boundary_mask(self, coord, segment, offset):
        device = coord.device
        n = coord.shape[0]
        C = self.num_classes

        scene = self._scene_index(n, offset, device)
        # Quantise to the radius-grid; shift spatial cells to start at 1 so that a
        # -1 neighbour stays >= 0 (keeps the linear key encoding collision-free).
        cell_xyz = torch.floor(coord[:, :3] / self.radius).long()
        cell_xyz = cell_xyz - cell_xyz.min(dim=0).values + 1
        cell = torch.cat([scene[:, None], cell_xyz], dim=1)              # (N, 4)
        spans = cell.max(dim=0).values + 3                              # +/-1 margin

        def encode(c):
            k = c[:, 0]
            k = k * spans[1] + c[:, 1]
            k = k * spans[2] + c[:, 2]
            k = k * spans[3] + c[:, 3]
            return k

        keys = encode(cell)
        uniq, inv = torch.unique(keys, sorted=True, return_inverse=True)  # (U,), (N,)
        U = uniq.shape[0]

        # Per-cell class presence (U, C) via a single amax scatter of one-hots.
        valid = (segment >= 0) & (segment < C)
        oh = F.one_hot(segment.clamp(0, C - 1), C).to(torch.float32)
        oh = oh * valid[:, None].to(oh.dtype)                           # drop ignore labels
        present = torch.zeros(U, C, dtype=torch.float32, device=device)
        present.scatter_reduce_(
            0, inv[:, None].expand(-1, C), oh, reduce="amax", include_self=True,
        )
        present = present > 0.5                                         # (U, C) bool
        own_oh = F.one_hot(segment.clamp(0, C - 1), C).bool() & valid[:, None]

        boundary = torch.zeros(n, dtype=torch.bool, device=device)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = cell.clone()
                    nb[:, 1] += dx
                    nb[:, 2] += dy
                    nb[:, 3] += dz
                    nk = encode(nb)
                    pos = torch.searchsorted(uniq, nk).clamp_(max=U - 1)
                    found = uniq[pos] == nk
                    has_other = (present[pos] & ~own_oh).any(dim=1)
                    boundary |= found & has_other
        return boundary & valid

    def update(self, outputs, batch):
        if torch.is_grad_enabled():
            return
        if "coord" not in batch:
            return
        coord = batch["coord"]
        segment = batch["segment"]
        offset = batch.get("offset", None)
        preds = torch.argmax(outputs, dim=1)
        bmask = self._boundary_mask(coord, segment, offset)
        if bmask.any():
            self.metric.update(preds[bmask], segment[bmask])

    def compute(self):
        if torch.is_grad_enabled():
            return {}
        iou = self.metric.compute()
        return {"boundary_mIoU": iou.mean().item()}


class DistillationMetrics:
    """Per-stage distillation tracker: total loss, per-stage loss, per-stage CKA.

    Reads from the Point dict produced by DistillerSegmentor.forward:
      * outputs["distill_loss"]:           scalar tensor (total weighted loss)
      * outputs["distill_per_stage"]:      (num_stages,) per-stage loss values
      * outputs["distill_cka_per_stage"]:  (num_stages,) per-stage linear CKA
                                            on the raw student/teacher features
                                            (no projection head) — better
                                            "is distillation working?" signal
                                            than the loss alone

    Returned keys (after .compute()):
      * "distill_loss":            running mean of total loss
      * "distill/stage{s}":        per-stage loss values (typically cosine)
      * "cka/stage{s}":            per-stage CKA in [0, 1]; higher = better
      * "cka/mean":                arithmetic mean of per-stage CKAs

    Plugs into MetricManager just like SegmentationMetrics (same update/compute
    interface) and stays a no-op gracefully when the model output is a plain
    tensor.
    """

    def __init__(self, num_stages: int, device: str = "cpu"):
        self.num_stages = num_stages
        self.device = device
        self.reset()

    def reset(self):
        self._per_stage_sum = torch.zeros(self.num_stages, device=self.device)
        self._per_stage_count = 0
        self._cka_sum = torch.zeros(self.num_stages, device=self.device)
        self._cka_count = 0
        self._total_sum = 0.0
        self._total_count = 0

    def update(self, outputs, batch):
        if not isinstance(outputs, dict):
            return
        if "distill_loss" in outputs.keys():
            v = outputs["distill_loss"]
            self._total_sum += float(v.detach().item() if torch.is_tensor(v) else v)
            self._total_count += 1
        if "distill_per_stage" in outputs.keys():
            ps = outputs["distill_per_stage"].detach().to(self.device).float()
            if ps.shape == self._per_stage_sum.shape:
                self._per_stage_sum += ps
                self._per_stage_count += 1
        if "distill_cka_per_stage" in outputs.keys():
            cka = outputs["distill_cka_per_stage"].detach().to(self.device).float()
            if cka.shape == self._cka_sum.shape:
                self._cka_sum += cka
                self._cka_count += 1

    def compute(self) -> dict:
        result = {}
        if self._total_count > 0:
            result["distill_loss"] = self._total_sum / self._total_count
        if self._per_stage_count > 0:
            avg = (self._per_stage_sum / self._per_stage_count).cpu().tolist()
            for s, v in enumerate(avg):
                result[f"distill/stage{s}"] = float(v)
        if self._cka_count > 0:
            cka_avg = (self._cka_sum / self._cka_count).cpu().tolist()
            for s, v in enumerate(cka_avg):
                result[f"cka/stage{s}"] = float(v)
            result["cka/mean"] = float(sum(cka_avg) / len(cka_avg))
        return result


def IoU(preds, ground_truth, n_classes):
    B, H, W = preds.shape

    iou = torch.empty(n_classes, dtype=torch.float32)
    for c in range(n_classes):
        p = preds==c
        g = ground_truth==c
        inter = (p*g).sum(dim=(1,2))
        union = p.sum(dim=(1,2)) + g.sum(dim=(1,2)) - inter
        iou_batch = (inter + eps) / (union + eps)
        iou[c] = iou_batch.mean()
    
    return iou


if __name__ == "__main__":
    
    target = torch.randint(0, 3, (10, 50, 50))
    preds = torch.randint(0, 3, (10, 50, 50))
    
    metric = MulticlassJaccardIndex(num_classes=3, average = None)
    iou = metric(preds, target)
    print(iou.shape)
    print(iou)
    print(iou.mean())

    iou = IoU(preds, target, 3)
    print(iou.shape)
    print(iou)