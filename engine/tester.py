"""
Evaluate a trained model on a test set.

Follows the structure of Pointcept's SemSegTester
(ditr/pointcept/engines/test.py): a per-fragment forward loop that scatters
softmax probabilities into an original-point accumulator via the batch
offsets, and a data_dict[0] passthrough collate for the TTA path. The
point-only fallback (_infer_single) is ours - Pointcept always voxelises into
fragment lists; we keep it so non-TTA configs still work.
"""

from __future__ import annotations

import json
import os
import time
from abc import abstractmethod
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from omegaconf import OmegaConf

from utils.config import Config
from utils.dist import init_dist, get_rank, get_world_size, get_local_rank
from metrics.metrics_manager import MetricManager


def _tta_passthrough_collate(batch):
    """Picklable passthrough collate for TTA mode (batch size 1 of result_dicts)."""
    return batch[0]


class _OrigIdxDataset(Dataset):
    """Tester-only wrapper that adds per-chunk orig_idx + n_las_pts to each sample.

    Makes the zone-aggregated tester read orig_idx in the parallel DataLoader
    workers (overlapped with the GPU) instead of blocking the main thread on a
    serial per-zone preload, which caused multi-minute 0%-GPU stalls at zone
    boundaries. Passthrough for datasets without get_chunk_meta (nuScenes, S3DIS).
    """

    def __init__(self, base):
        self.base = base
        self._has_meta = hasattr(base, "get_chunk_meta")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self._has_meta and isinstance(sample, dict):
            meta = self.base.get_chunk_meta(idx)
            oi = meta.get("orig_idx")
            # int32 halves the array vs the on-disk int64 (LAS clouds < 2**31 pts);
            # the .long() upcast for GPU indexing later is transparent.
            sample["orig_idx"] = oi.astype(np.int32, copy=False) if oi is not None else None
            sample["n_las_pts"] = meta.get("n_las_pts")
        return sample


class _IndexSampler(Sampler):
    """Yields a fixed, explicit list of dataset indices in order.

    Used both to group chunks by zone (zone-aggregated eval) and to hand each
    DDP rank its own disjoint shard of chunk indices.
    """

    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class BaseTester:
    """Task-agnostic inference loop over a test split.

    Owns the generic plumbing: DDP, device and output dirs, the per-sample
    loop with on-disk resume and crash-retry markers, and metric reporting.
    Subclasses provide the model, dataloader, metrics and per-sample inference
    through the abstract methods, plus the optional hooks at the bottom.
    """

    def __init__(self, cfg=None):
        self._raw_cfg = cfg
        self.cfg = Config(cfg)

        # Distributed setup (torchrun) - must precede device & dir setup so each
        # rank binds its own GPU and only rank 0 creates output dirs. No-op for
        # a plain python launch. Mirrors the trainer.
        init_dist()
        self.rank       = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        self.is_ddp     = self.world_size > 1

        if self.is_ddp:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(
                getattr(self.cfg.training, "device", "cuda" if torch.cuda.is_available() else "cpu")
            )

        self.save_dir = Path(self.cfg.outputs.save_dir)

        _test_cfg = getattr(cfg, "test", None)
        _results_dir = getattr(_test_cfg, "results_dir", None) if _test_cfg is not None else None
        self.results_dir = Path(_results_dir) if _results_dir else None
        # Retry budget for hard per-chunk failures. A CUDA "illegal memory
        # access" kills the whole process, so each chunk writes an on-disk
        # attempt marker BEFORE inference; on a re-launch, a chunk that already
        # burned this many attempts without a result is skipped and logged to
        # failed_chunks.txt. 0 disables the net.
        self._max_chunk_retries = (
            int(getattr(_test_cfg, "max_chunk_retries", 1)) if _test_cfg is not None else 1
        )
        if self.results_dir is not None and self.rank == 0:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.is_ddp:
            dist.barrier()

        self.amp = getattr(self.cfg.training, "amp", False)
        _dtype_str = getattr(self.cfg.training, "amp_dtype", "float16")
        self.amp_dtype = getattr(torch, _dtype_str)

        self.model = None
        self.dataset = None
        self.test_loader = None
        self.metric_manager = None
        self._test_indices = None  # this rank's chunk shard, set in get_dataloader
        # True when this session skipped already-saved work (a resume); only then
        # does _finalize_metrics need to rebuild the metric from disk.
        self._resumed = False

        self.setup_test()

    # Setup

    def setup_test(self):
        self.setup_model()
        if self.model is None:
            raise RuntimeError("setup_model() must define self.model")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.test_loader, self.dataset = self.get_dataloader()
        self.metric_manager = self.setup_metrics()

    # Inference loop

    @torch.no_grad()
    def test(self):
        """Run inference over this rank's shard of the test split and save results.

        Under torchrun the chunks are sharded across ranks (strided, disjoint,
        no padding) via self._test_indices; each rank writes only its own
        chunks' .npy files and the metric is reduced across ranks at the end.
        Single-GPU behaviour is unchanged (the shard is the whole split).
        """
        indices = self._test_indices
        n = len(indices)
        ran_metrics = False
        n_skipped = 0
        n_failed = 0

        # ci is the true dataset index (needed for naming / chunk_paths); k is
        # this rank's local progress counter.
        for k, (ci, sample) in enumerate(zip(indices, self.test_loader)):
            chunk_name = self._chunk_name(ci)
            area = self._chunk_area(ci)
            label = f"{area}/{chunk_name}" if area else chunk_name

            attempt_path = None
            if self.results_dir is not None:
                out_path = (self.results_dir / area / f"{chunk_name}.npy"
                            if area else self.results_dir / f"{chunk_name}.npy")
                if out_path.exists():
                    n_skipped += 1
                    print(f"  [r{self.rank} {k + 1}/{n}] skip (exists) -> {label}.npy")
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Record the attempt on disk (flushed + fsync'd) BEFORE inference;
                # once a chunk has burned its retry budget without producing a
                # result, skip it for good.
                if self._max_chunk_retries > 0:
                    attempt_path = out_path.with_suffix(".attempt")
                    n_prev = 0
                    if attempt_path.exists():
                        try:
                            n_prev = int(attempt_path.read_text().strip() or "0")
                        except ValueError:
                            n_prev = 0
                    if n_prev >= self._max_chunk_retries:
                        n_failed += 1
                        print(f"  [r{self.rank} {k + 1}/{n}] SKIP (failed {n_prev}x, "
                              f"giving up) -> {label}", flush=True)
                        self._record_failed_chunk(label, n_prev)
                        continue
                    with open(attempt_path, "w") as f:
                        f.write(str(n_prev + 1))
                        f.flush()
                        os.fsync(f.fileno())

            pred_all, did_metric = self._infer_sample(sample)
            ran_metrics = ran_metrics or did_metric

            if self.results_dir is not None:
                np.save(out_path, pred_all.cpu().numpy().astype(np.int16))
                # Success: drop the attempt marker so a future rerun treats the
                # chunk as cleanly done (the .npy is the source of truth anyway).
                if attempt_path is not None and attempt_path.exists():
                    attempt_path.unlink()
                print(f"  [r{self.rank} {k + 1}/{n}] saved -> {label}.npy")
            else:
                print(f"  [r{self.rank} {k + 1}/{n}] done")

        if n_skipped:
            print(f"[Tester r{self.rank}] Skipped {n_skipped} already-completed chunks.")
        if n_failed:
            print(f"[Tester r{self.rank}] Gave up on {n_failed} chunk(s) after exhausting "
                  f"retries (see failed_chunks.txt).")
        if n_skipped or n_failed:
            self._resumed = True

        self._write_manifest()
        self._maybe_report_metrics(ran_metrics)

        # Authoritative metric after a resume: rebuilt from the saved predictions
        # on disk (rank 0, after the barrier in _maybe_report_metrics).
        self._finalize_metrics()
        # Rank 0 may have done a long rank-0-only recompute above; rendezvous so
        # no rank races ahead to process-group teardown.
        if self.is_ddp:
            dist.barrier()

    # Utilities

    def _chunk_name(self, batch_idx: int) -> str:
        if hasattr(self.dataset, "chunk_paths"):
            return Path(self.dataset.chunk_paths[batch_idx]).name
        return f"chunk_{batch_idx:06d}"

    def _chunk_area(self, batch_idx: int) -> str | None:
        """Return the area/zone name (parent directory) for a chunk, or None."""
        if hasattr(self.dataset, "chunk_paths"):
            return Path(self.dataset.chunk_paths[batch_idx]).parent.name
        return None

    def _record_failed_chunk(self, label: str, n_attempts: int) -> None:
        """Append a permanently-skipped chunk to results_dir/failed_chunks.txt."""
        if self.results_dir is None:
            return
        try:
            with open(self.results_dir / "failed_chunks.txt", "a") as f:
                f.write(f"{label}\tattempts={n_attempts}\trank={self.rank}\n")
        except OSError:
            pass

    def _move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x) for x in batch)
        return batch

    def _report_metrics(self, metrics: dict):
        lines = ["=" * 60, "Test Metrics", "=" * 60]
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    lines.append(f"  {k:30s}: {v.item():.4f}")
                else:
                    vals = ", ".join(f"{x:.4f}" for x in v.flatten().tolist())
                    lines.append(f"  {k:30s}: [{vals}]")
            elif isinstance(v, (int, float)):
                lines.append(f"  {k:30s}: {v:.4f}")
        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)
        if self.results_dir is not None:
            out = self.results_dir / "metrics.txt"
            out.write_text(report)
            print(f"[Tester] Metrics written to {out}")

    def _maybe_report_metrics(self, ran_metrics: bool):
        """Finalize metrics across ranks and report on rank 0.

        MetricManager.compute() is a collective under DDP (per-rank confusion
        matrices are summed), so ran_metrics is reduced first so every rank
        agrees on whether to call it. Only rank 0 prints / writes the report.
        """
        if self.is_ddp:
            flag = torch.tensor([1 if ran_metrics else 0], device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            ran_metrics = bool(flag.item())

        if ran_metrics:
            results = self.metric_manager.compute()  # collective under DDP
            if self.rank == 0:
                self._report_metrics(results)
        elif self.rank == 0:
            if self.results_dir is not None:
                print(f"[Tester] Predictions saved to {self.results_dir}")
            else:
                print("[Tester] No labels and no results_dir - nothing saved.")

        if self.is_ddp:
            dist.barrier()

    # Abstract interface

    @abstractmethod
    def setup_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_dataloader(self):
        """Return (DataLoader, dataset); must also set self._test_indices."""
        raise NotImplementedError

    @abstractmethod
    def setup_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def _infer_sample(self, sample):
        """Run inference on one test sample.

        Returns:
            (pred, did_metric): per-point predictions to save, and whether the
            metric manager was updated for this sample.
        """
        raise NotImplementedError

    # Optional hooks

    def _write_manifest(self):
        """Write a manifest of the split's samples to results_dir (rank 0)."""
        pass

    def _finalize_metrics(self):
        """Recompute the authoritative metric after a resumed run."""
        pass


# Concrete implementation: point cloud segmentation

class SemSegTester(BaseTester):
    """Semantic segmentation tester: TTA fragments, zone aggregation, resume."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # Reuse the image ViT across a chunk's TTA fragments (4-5x faster
        # image-fusion testing). Set test.image_feat_cache: false to disable.
        _test_cfg = getattr(self._raw_cfg, "test", None)
        self._img_feat_cache_enabled = (
            bool(getattr(_test_cfg, "image_feat_cache", True)) if _test_cfg is not None else True
        )

    def setup_model(self):
        from models import MODELS
        model_cfg = OmegaConf.to_container(self._raw_cfg.model, resolve=True)
        self.model = MODELS.build(model_cfg)

        _test_cfg = getattr(self._raw_cfg, "test", None)

        # Cap how many cameras go through the image ViT per forward: a single
        # full-resolution test chunk can pack far more cameras than any training
        # step, OOMing the ViT. Configure via test.vit_cam_batch (default 8,
        # <=0 / null disables); train/val keep the model default.
        _vit_cam_batch = getattr(_test_cfg, "vit_cam_batch", 8) if _test_cfg is not None else 8
        if _vit_cam_batch and _vit_cam_batch > 0:
            for m in self.model.modules():
                if hasattr(m, "image_cam_batch"):
                    m.image_cam_batch = int(_vit_cam_batch)

        # Use spconv's Native (gather-GEMM-scatter) algorithm for inference.
        # The default implicit_gemm intermittently raises CUDA error 700 on some
        # sparse structures that TTA fragments expose; Native returns the same
        # result through a plain cuBLAS GEMM. Tester-only, so training keeps the
        # faster default.
        import spconv.pytorch as spconv
        from spconv.core import ConvAlgo
        n_conv = 0
        for m in self.model.modules():
            if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d, spconv.SparseInverseConv3d)):
                m.algo = ConvAlgo.Native
                n_conv += 1
        if self.rank == 0:
            print(f"[Tester] spconv: using Native algo on {n_conv} conv(s) for robust inference")

        ckpt_path = getattr(_test_cfg, "checkpoint", None) if _test_cfg is not None else None
        if ckpt_path is None:
            ckpt_path = self.save_dir / "weights" / "best.pt"
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_state = ckpt["model"]
        # Strip "module." prefix from checkpoints saved without DDP unwrapping
        if any(k.startswith("module.") for k in raw_state):
            raw_state = {k.removeprefix("module."): v for k, v in raw_state.items()}
        # Reconcile torch.compile's "._orig_mod." key prefix in either direction
        # (compiled checkpoint + eager model, or the reverse) by matching keys on
        # their compile-stripped names.
        def _strip_oc(k):
            return k.replace("._orig_mod.", ".")
        _model_by_stripped = {_strip_oc(k): k for k in self.model.state_dict()}
        raw_state = {
            _model_by_stripped.get(_strip_oc(k), k): v for k, v in raw_state.items()
        }
        missing, unexpected = self.model.load_state_dict(raw_state, strict=False)
        if missing:
            print(f"[Tester] {len(missing)} missing keys (new layers randomly initialised)")
            for k in missing[:5]:
                print(f"  - {k}")
        if unexpected:
            print(f"[Tester] {len(unexpected)} unexpected keys (ignored)")
        if not missing and not unexpected:
            print(f"[Tester] Weights loaded from {ckpt_path}")

    def setup_metrics(self):
        from metrics.metrics import SegmentationMetrics
        return MetricManager([SegmentationMetrics(
            num_classes=self.cfg.data.num_classes,
            ignore_index=self.cfg.data.ignore_index,
            device=self.device,
        )])

    @torch.no_grad()
    def test(self):
        """Dispatch to zone-aggregated or per-chunk evaluation.

        Zone-aggregated mode is selected automatically when the dataset exposes
        get_chunk_meta and the first chunk has orig_idx (Atocha / A9CR); other
        datasets fall through to the per-chunk BaseTester.test(). Force with
        test.zone_aggregated: true/false.
        """
        use_zone = False
        if hasattr(self.dataset, "get_chunk_meta"):
            try:
                meta = self.dataset.get_chunk_meta(0)
                use_zone = (meta["orig_idx"] is not None)
            except Exception:
                pass
        _test_cfg = getattr(self._raw_cfg, "test", None)
        if _test_cfg is not None:
            use_zone = getattr(_test_cfg, "zone_aggregated", use_zone)

        if use_zone:
            print("[Tester] Using zone-aggregated evaluation (unbiased mIoU for overlapping chunks).")
            return self._test_zone_aggregated()

        print("[Tester] Using per-chunk evaluation.")
        return super().test()

    # Inference

    def _infer_sample(self, sample):
        if isinstance(sample, dict) and "fragment_list" in sample:
            return self._infer_tta(sample)
        return self._infer_single(sample)

    def _tta_accumulate(self, sample):
        """Accumulate softmax probs over all TTA fragments of one chunk.

        Each fragment is collated alone, forwarded, and its softmax scattered
        into a per-original-point accumulator via the fragment's index/offset
        (the Pointcept pattern). The shared image tensor is re-attached to every
        fragment, and one _img_cache_key per chunk lets image-fusion backbones
        run the ViT once and reuse it; models that don't read it ignore it.

        Args:
            sample: TTA dict with fragment_list, n_orig and optional shared_image.

        Returns:
            (n_orig, num_classes) float32 prob accumulator on the device.
        """
        from datasets.utils import point_collate_fn

        num_classes = self.cfg.data.num_classes
        pred = torch.zeros(sample["n_orig"], num_classes, device=self.device, dtype=torch.float32)
        shared_image = sample.get("shared_image")
        self._chunk_fwd_id = getattr(self, "_chunk_fwd_id", 0) + 1
        for frag in sample["fragment_list"]:
            if shared_image is not None:
                frag = {**frag, "image": shared_image}
            input_dict = point_collate_fn([frag], mix_prob=0)
            input_dict = self._move_to_device(input_dict)
            if self._img_feat_cache_enabled:
                input_dict["_img_cache_key"] = self._chunk_fwd_id
            with torch.autocast(self.device.type, enabled=self.amp, dtype=self.amp_dtype):
                outputs = self.model(input_dict)
            pred_part = F.softmax(self._get_logits(outputs).float(), dim=-1)
            idx_part = input_dict["index"].view(-1).long()
            bs = 0
            for be in input_dict["offset"].tolist():
                pred[idx_part[bs:be]] += pred_part[bs:be]
                bs = be
        return pred

    def _infer_tta(self, sample):
        """Fragment inference for one chunk: accumulate, argmax, expand, score.

        Args:
            sample: TTA dict with fragment_list, n_orig, optional inverse /
                origin_segment / segment.

        Returns:
            (pred_all, did_metric): per-point label predictions at the finest
            available resolution, and whether metrics were updated.
        """
        pred = self._tta_accumulate(sample)
        pred_all = pred.argmax(dim=1)

        # Expand from pre-vox resolution back to full scanner resolution when
        # the test transform included a GridSample with return_inverse=True.
        if "inverse" in sample:
            inv = sample["inverse"]
            if not isinstance(inv, torch.Tensor):
                inv = torch.as_tensor(inv)
            pred_all = pred_all[inv.to(self.device)]

        did_metric = False
        if "origin_segment" in sample:
            # Full-resolution scoring (needs Copy(segment -> origin_segment)
            # before the pre-vox GridSample in test.transform).
            target = sample["origin_segment"]
            if not isinstance(target, torch.Tensor):
                target = torch.as_tensor(target)
            target = target.to(self.device).long()
            inv = sample["inverse"]
            if not isinstance(inv, torch.Tensor):
                inv = torch.as_tensor(inv)
            expanded_pred = pred[inv.to(self.device)]  # (N_raw, C)
            self.metric_manager.update(expanded_pred, {"segment": target})
            did_metric = True
        elif "segment" in sample:
            # Labels only at the pre-vox representative level - fine for a
            # sanity check.
            target = sample["segment"]
            if not isinstance(target, torch.Tensor):
                target = torch.as_tensor(target)
            target = target.to(self.device).long()
            self.metric_manager.update(pred, {"segment": target})
            did_metric = True

        return pred_all, did_metric

    def _infer_single(self, batch):
        batch = self._move_to_device(batch)

        with torch.autocast(self.device.type, enabled=self.amp, dtype=self.amp_dtype):
            outputs = self.model(batch)

        logits = self._get_logits(outputs)       # (N_vox, C)
        pred_vox = torch.argmax(logits, dim=1)   # (N_vox,)

        # Expand voxelised predictions to all original points via the GridSample
        # inverse map when present.
        if "inverse" in batch:
            pred_all = pred_vox[batch["inverse"]]
        else:
            pred_all = pred_vox

        did_metric = False
        if "segment" in batch:
            self.metric_manager.update(logits, batch)
            did_metric = True

        return pred_all, did_metric

    @torch.no_grad()
    def _infer_chunk_probs_and_labels(self, sample):
        """Return (probs, labels) at raw chunk-point resolution for zone aggregation.

        Raw chunk-point level means the points stored in xyz.npy
        (post-preprocessing-vox, pre-inference-vox). TTA fragment probs, or
        non-TTA voxel probs, are expanded via the GridSample inverse when the
        test transform stored one.

        Args:
            sample: a TTA dict (fragment_list) or a collated non-TTA batch.

        Returns:
            probs: (N_chunk, C) float32 tensor on the device.
            labels: (N_chunk,) long tensor, or None if no labels present.
        """
        if isinstance(sample, dict) and "fragment_list" in sample:
            # TTA path
            pred = self._tta_accumulate(sample)
            if "inverse" in sample:
                inv = torch.as_tensor(sample["inverse"]).to(self.device).long()
                probs = pred[inv]   # (N_raw, C)
                if "origin_segment" in sample:
                    labels = torch.as_tensor(sample["origin_segment"]).to(self.device).long()
                elif "segment" in sample:
                    labels = torch.as_tensor(sample["segment"]).to(self.device).long()[inv]
                else:
                    labels = None
            else:
                probs  = pred   # (N_orig == N_chunk, C)
                labels = (torch.as_tensor(sample["segment"]).to(self.device).long()
                          if "segment" in sample else None)

        else:
            # Non-TTA path. Requires GridSample(return_inverse=True) in the test
            # transform so inference-vox probs can be expanded to chunk level.
            batch = self._move_to_device(sample)
            with torch.autocast(self.device.type, enabled=self.amp, dtype=self.amp_dtype):
                outputs = self.model(batch)
            logits   = self._get_logits(outputs)
            probs_vox = F.softmax(logits.float(), dim=-1)   # (N_vox, C)

            if "inverse" in batch:
                inv    = batch["inverse"].long()
                probs  = probs_vox[inv]                      # (N_chunk, C)
                labels = (batch["segment"][inv].long()
                          if "segment" in batch else None)
            else:
                probs = probs_vox
                seg   = batch.get("segment")
                labels = (seg.long() if isinstance(seg, torch.Tensor)
                          else torch.as_tensor(seg).to(self.device).long()
                          if seg is not None else None)

        return probs, labels

    # Zone-aggregated evaluation

    @torch.no_grad()
    def _test_zone_aggregated(self):
        """Proper mIoU for overlapping-chunk test sets (Atocha / A9CR).

        Per zone (one LAS file / scan): build the zone -> chunk map, shard whole
        zones across ranks, stream the chunks through a DataLoader (zone-grouped
        order, workers loading orig_idx too), accumulate probs into a per-zone
        GPU buffer sized from n_las_pts, and on zone change flush: argmax, save
        per-chunk .npy, update metrics. orig_idx maps chunk points to rows of
        the post-removal, pre-preprocessing-vox LAS cloud.
        """
        from collections import defaultdict

        num_classes  = self.cfg.data.num_classes
        ignore_index = self.cfg.data.ignore_index

        # Phase A: build the zone -> chunk map and shard whole zones across
        # ranks (a zone's chunks must stay on one rank so its accumulator is
        # complete). Strided, disjoint, no duplication.
        n = len(self.dataset)
        zone_to_chunks: dict = defaultdict(list)
        for i in range(n):
            zone = self._chunk_area(i) or "default_zone"
            zone_to_chunks[zone].append(i)

        all_zones = list(zone_to_chunks.keys())
        my_zones = all_zones[self.rank::self.world_size]

        # Resume by default: skip whole zones whose chunks are all already saved.
        # The in-run metric then covers only this session's zones, but
        # _finalize_metrics recomputes the complete mIoU from disk at the end.
        # Set test.resume: false to force a full recompute.
        _tcfg = getattr(self._raw_cfg, "test", None)
        _resume = bool(getattr(_tcfg, "resume", True)) if _tcfg is not None else True

        def _chunk_out_path(ci):
            area = self._chunk_area(ci)
            cname = self._chunk_name(ci)
            return (self.results_dir / area / f"{cname}.npy"
                    if area else self.results_dir / f"{cname}.npy")

        if _resume and self.results_dir is not None:
            pending, n_skip_zones = [], 0
            for z in my_zones:
                if all(_chunk_out_path(ci).exists() for ci in zone_to_chunks[z]):
                    n_skip_zones += 1
                else:
                    pending.append(z)
            if n_skip_zones:
                print(f"[Tester r{self.rank}] Resume: skipping {n_skip_zones} fully-saved "
                      f"zone(s); {len(pending)} remaining.", flush=True)
                self._resumed = True
            my_zones = pending

        ordered_indices = [ci for z in my_zones for ci in zone_to_chunks[z]]
        print(
            f"[Tester r{self.rank}] Zone-aggregated evaluation: "
            f"{len(my_zones)}/{len(all_zones)} zone(s), "
            f"{len(ordered_indices)} chunks on this rank"
        )

        # Phase B: streaming DataLoader. orig_idx / n_las_pts are loaded by the
        # workers (_OrigIdxDataset), overlapped with the GPU; only the current
        # zone's arrays stay resident (in current_oi) for the .npy save.
        loader = self._make_test_loader(
            _OrigIdxDataset(self.dataset), ordered_indices, pin_memory=True
        )

        # Phase C: stream + flush-on-zone-change
        current_zone = None
        current_oi: dict = {}   # orig_idx arrays for the current zone only
        zone_pred = None        # GPU
        zone_labels = None      # GPU
        has_labels = False
        chunks_in_zone: list = []
        ran_metrics = False

        def _flush(zone, pred, labels, lbls_seen, chunk_idxs, oi_map):
            nonlocal ran_metrics
            if pred is None:
                return
            zone_argmax = pred.argmax(dim=1).cpu()
            if self.results_dir is not None:
                for ci in chunk_idxs:
                    oi = oi_map.get(ci)
                    if oi is None:
                        continue
                    chunk_pred = zone_argmax[torch.from_numpy(oi).long()]
                    area = self._chunk_area(ci)
                    cname = self._chunk_name(ci)
                    out_path = (self.results_dir / area / f"{cname}.npy"
                                if area else self.results_dir / f"{cname}.npy")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, chunk_pred.numpy().astype(np.int16))
                print(f"  Zone {zone!r}: {len(chunk_idxs)} chunk prediction(s) saved", flush=True)
            if lbls_seen:
                self.metric_manager.update(pred, {"segment": labels})
                ran_metrics = True

        # Per-chunk timing so a slow or stalled inference is visible in the log:
        # infer is the CUDA-synced GPU forward, wait is time spent on the
        # DataLoader (CPU load / JPEG decode / TTA voxelisation).
        n_total = len(ordered_indices)
        n_done = 0
        t_loop0 = time.perf_counter()
        t_prev = t_loop0

        for ci, sample in zip(ordered_indices, loader):
            data_wait = time.perf_counter() - t_prev
            zone = self._chunk_area(ci) or "default_zone"
            oi = sample.get("orig_idx") if isinstance(sample, dict) else None
            n_las_pts = sample.get("n_las_pts") if isinstance(sample, dict) else None

            if zone != current_zone:
                _flush(current_zone, zone_pred, zone_labels, has_labels,
                       chunks_in_zone, current_oi)
                current_zone = zone
                current_oi = {}
                chunks_in_zone = []
                has_labels = False
                if oi is None or n_las_pts is None:
                    zone_pred = None
                    zone_labels = None
                    print(f"  Zone {zone!r}: no orig_idx.npy found - skip", flush=True)
                else:
                    # Size from n_las_pts (post-removal cloud size), an upper
                    # bound on max(orig_idx)+1. Rows never written stay
                    # ignore_index, so they don't affect the metric.
                    acc_size = int(n_las_pts)
                    zone_pred = torch.zeros(
                        acc_size, num_classes, dtype=torch.float32, device=self.device
                    )
                    zone_labels = torch.full(
                        (acc_size,), fill_value=ignore_index,
                        dtype=torch.long, device=self.device,
                    )

            if oi is None or zone_pred is None:
                t_prev = time.perf_counter()
                continue

            # Pre-inference marker so a stall is attributable: fast print + big
            # wait = DataLoader bottleneck; fast print then silence = model hang.
            n_frag = len(sample["fragment_list"]) if isinstance(sample, dict) and "fragment_list" in sample else 1
            print(f"  [r{self.rank} {n_done+1}/{n_total}] {zone}/{self._chunk_name(ci)}  "
                  f"wait={data_wait:.1f}s - inferring ({n_frag} TTA frag)...", flush=True)

            t_infer0 = time.perf_counter()
            probs, labels = self._infer_chunk_probs_and_labels(sample)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t_infer = time.perf_counter() - t_infer0

            current_oi[ci] = oi   # retained for the per-chunk .npy save in _flush
            orig_idx_gpu = torch.from_numpy(oi).long().to(self.device, non_blocking=True)
            zone_pred[orig_idx_gpu] += probs
            if labels is not None:
                zone_labels[orig_idx_gpu] = labels
                has_labels = True
            chunks_in_zone.append(ci)

            n_done += 1
            elapsed = time.perf_counter() - t_loop0
            avg = elapsed / n_done
            eta_min = (n_total - n_done) * avg / 60.0
            print(f"      -> infer={t_infer:.2f}s  avg={avg:.2f}s/chunk  ETA={eta_min:.1f}min", flush=True)
            t_prev = time.perf_counter()

        _flush(current_zone, zone_pred, zone_labels, has_labels,
               chunks_in_zone, current_oi)

        total_t = time.perf_counter() - t_loop0
        print(
            f"[Tester r{self.rank}] Done: {n_done} chunk(s) inferred in {total_t/60:.1f} min "
            f"({total_t/max(n_done,1):.2f}s/chunk avg)",
            flush=True,
        )

        self._write_manifest()
        self._maybe_report_metrics(ran_metrics)

        # Authoritative, resume-safe metric from every saved prediction (rank 0,
        # after the barrier above). Skipped for a fresh run, where the in-run
        # zone-aggregated metric is already complete.
        self._finalize_metrics()
        # Rendezvous before teardown so a long rank-0-only recompute can't leave
        # other ranks arriving at process-group destroy unevenly.
        if self.is_ddp:
            dist.barrier()

    # Metrics

    @torch.no_grad()
    def _finalize_metrics(self):
        """Rebuild the complete test metric from the per-chunk .npy files on disk.

        Only a resumed run needs this: its in-run metric covers just the chunks
        processed this session, while the saved predictions cover the whole
        split across all runs. The saved pred is the per-point argmax at the
        same resolution as get_data(idx)["segment"], so they align
        point-for-point. All ranks enter (the resumed-flag all_reduce is a
        collective); the recompute itself runs on rank 0 only, with no
        cross-rank collectives (a hand-accumulated confusion matrix, identical
        to SegmentationMetrics, avoids torchmetrics' compute() sync which would
        deadlock against the post-finalize barrier).
        """
        resumed = self._resumed
        if self.is_ddp:
            flag = torch.tensor([1 if resumed else 0], device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            resumed = bool(flag.item())
        if self.rank != 0 or self.results_dir is None:
            return
        if not resumed:
            print("[Tester] Fresh run - every chunk was scored in-line; skipping the "
                  "redundant from-disk metric recompute.")
            return
        dataset = self.dataset
        if dataset is None or not hasattr(dataset, "chunk_paths"):
            return
        num_classes = self.cfg.data.num_classes
        ignore_index = self.cfg.data.ignore_index
        confmat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)

        # Labels don't need images - skip JPEG decode for a fast GT-only pass.
        had_image = getattr(dataset, "with_image", None)
        if had_image is not None:
            dataset.with_image = False

        n_used = n_missing = n_nolabel = n_mismatch = 0
        n_chunks = len(dataset.chunk_paths)

        # The cost is reading every chunk's GT + prediction off a (likely
        # network) filesystem - pure I/O latency. Overlap it with a thread pool:
        # workers load (gt, pred), the main thread does the serial GPU metric
        # update. get_data is read-only, so threads on the shared dataset are
        # safe; ThreadPoolExecutor.map preserves order.
        from concurrent.futures import ThreadPoolExecutor

        def _load_chunk(ci):
            name, area = self._chunk_name(ci), self._chunk_area(ci)
            out_path = (self.results_dir / area / f"{name}.npy"
                        if area else self.results_dir / f"{name}.npy")
            if not out_path.exists():
                return "missing", None, None
            try:
                gt = dataset.get_data(ci).get("segment")
            except Exception:
                return "nolabel", None, None
            if gt is None:
                return "nolabel", None, None
            pred = np.load(out_path)
            gt = np.asarray(gt)
            if pred.shape[0] != gt.shape[0]:
                return "mismatch", None, None
            return "ok", gt, pred

        # I/O-bound, so threads can exceed core count; floor at 8 for decent
        # network-FS overlap.
        _tcfg = getattr(self._raw_cfg, "test", None)
        n_io_workers = max(8, int(getattr(_tcfg, "num_workers", 8) or 8) if _tcfg else 8)
        print(f"[Tester] Resumed run - recomputing the complete metric from "
              f"{n_chunks:,} saved chunk prediction(s) on disk "
              f"({n_io_workers}-way parallel I/O)...", flush=True)
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=n_io_workers) as ex:
                for k, (status, gt, pred) in enumerate(ex.map(_load_chunk, range(n_chunks))):
                    if k and k % 500 == 0:
                        el = time.perf_counter() - t0
                        print(f"[Tester]   ...scored {k:,}/{n_chunks:,} chunks "
                              f"({el:.0f}s, {k / max(el, 1e-9):.0f} chunk/s)", flush=True)
                    if status == "missing":
                        n_missing += 1
                        continue
                    if status == "nolabel":
                        n_nolabel += 1
                        continue
                    if status == "mismatch":
                        n_mismatch += 1
                        continue
                    pred_t = torch.as_tensor(pred, device=self.device).long().clamp_(0, num_classes - 1)
                    gt_t = torch.as_tensor(gt, device=self.device).long()
                    valid = (gt_t >= 0) & (gt_t < num_classes) & (gt_t != ignore_index)
                    confmat += torch.bincount(
                        gt_t[valid] * num_classes + pred_t[valid],
                        minlength=num_classes * num_classes,
                    ).reshape(num_classes, num_classes)
                    n_used += 1
        finally:
            if had_image is not None:
                dataset.with_image = had_image
        print(f"[Tester]   loaded+scored {n_used:,} chunk(s) in "
              f"{time.perf_counter() - t0:.0f}s; computing final metric...", flush=True)

        if n_used == 0:
            if n_nolabel and not n_mismatch:
                print("[Tester] Complete metrics: test split has no labels - predictions saved only.")
            else:
                print(f"[Tester] Complete metrics: nothing to score "
                      f"(missing={n_missing}, no-label={n_nolabel}, mismatch={n_mismatch}).")
            return

        results = self._metrics_from_confmat(confmat)
        print(f"\n[Tester] ===== COMPLETE test metrics - {n_used} chunk(s) from saved "
              f"predictions (resume-safe) =====")
        if n_missing or n_mismatch:
            print(f"[Tester]   note: {n_missing} chunk(s) without a saved prediction, "
                  f"{n_mismatch} skipped for length mismatch.")
        self._report_metrics(results)   # prints + overwrites metrics.txt with the complete result

    @staticmethod
    def _metrics_from_confmat(confmat: torch.Tensor) -> dict:
        """Per-class IoU / recall / precision from a (C, C) confusion matrix.

        Rows are ground truth, cols are prediction. Same keys and formulas as
        SegmentationMetrics.compute(), but fully local (no torchmetrics, no DDP
        sync). Classes with no support get 0 where torchmetrics would give NaN.

        Args:
            confmat: (C, C) integer confusion matrix.

        Returns:
            dict with accuracy / iou / precision per class, their means, OA and
            the confusion matrix itself.
        """
        cm = confmat.double()
        tp = cm.diag()
        gt_sum = cm.sum(dim=1)        # support per class (target rows)
        pred_sum = cm.sum(dim=0)      # predicted per class (cols)
        union = gt_sum + pred_sum - tp
        iou = tp / union.clamp(min=1)
        recall = tp / gt_sum.clamp(min=1)        # = Accuracy(average=None) per class
        precision = tp / pred_sum.clamp(min=1)
        total = cm.sum()
        return {
            "accuracy": recall.float().cpu(),
            "mean_accuracy": recall.mean().item(),
            "OA": (tp.sum() / total.clamp(min=1)).item(),
            "iou": iou.float().cpu(),
            "mIoU": iou.mean().item(),
            "precision": precision.float().cpu(),
            "mean_precision": precision.mean().item(),
            "confusion_matrix": confmat.long().cpu(),
        }

    # Data plumbing

    def _get_logits(self, outputs):
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        return outputs

    def _write_manifest(self):
        """Write chunk_manifest.json mapping "{area}/{chunk}" to its source path.

        The key is unique across areas even when chunk directory names repeat
        (e.g. chunk_00000 in every area). Covers the whole split, so rank 0 only.
        """
        if (self.rank != 0 or self.results_dir is None
                or not hasattr(self.dataset, "chunk_paths")):
            return
        manifest = {
            f"{Path(p).parent.name}/{Path(p).name}": str(p)
            for p in self.dataset.chunk_paths
        }
        manifest_path = self.results_dir / "chunk_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[Tester] Chunk manifest saved -> {manifest_path}")

    def _make_test_loader(self, dataset, indices, pin_memory):
        """DataLoader over an explicit chunk-index list, sized from the test config.

        Each test sample is a full multi-aug TTA fragment list (roughly n_augs
        voxelised copies of the chunk), and num_workers x prefetch_factor of
        them sit in host RAM at once - so workers/prefetch come from test.*
        (prefetch defaults to 1), falling back to training only if test.* is
        unset. Workers are not persistent: the loader is consumed once, and
        letting them die with the loop avoids the exit-join hang.

        Args:
            dataset: the dataset (possibly wrapped in _OrigIdxDataset).
            indices: explicit dataset indices to iterate, in order.
            pin_memory: passed through to the DataLoader.

        Returns:
            the DataLoader.
        """
        from datasets.utils import point_collate_fn

        _loader_cfg = getattr(self._raw_cfg, "test", None) or self.cfg.training
        num_workers = getattr(_loader_cfg, "num_workers", 0)
        prefetch_factor = (
            getattr(_loader_cfg, "prefetch_factor", 1) if num_workers > 0 else None
        )
        base = dataset.base if isinstance(dataset, _OrigIdxDataset) else dataset
        if getattr(base, "test_mode", False):
            collate = _tta_passthrough_collate
        else:
            collate = partial(point_collate_fn, mix_prob=0)
        return DataLoader(
            dataset,
            batch_size=1,
            sampler=_IndexSampler(indices),
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            persistent_workers=False,
            prefetch_factor=prefetch_factor,
        )

    def get_dataloader(self):
        dataset = self._build_test_dataset()
        # Shard chunks across ranks: strided, disjoint, no padding/duplication
        # (so the summed per-rank confusion matrices equal the single-GPU
        # result). Single-GPU: the whole split in order.
        self._test_indices = list(range(self.rank, len(dataset), self.world_size))

        # Debug skip-ahead: test.start_at (chunk name or global index) jumps
        # straight to a chunk of interest without loading anything before it.
        # Leave unset for a normal full run.
        _start_at = getattr(getattr(self._raw_cfg, "test", None), "start_at", None)
        if _start_at is not None and str(_start_at) != "":
            start_idx = None
            if isinstance(_start_at, int) or str(_start_at).isdigit():
                start_idx = int(_start_at)
            elif hasattr(dataset, "chunk_paths"):
                names = [Path(p).name for p in dataset.chunk_paths]
                for i, nm in enumerate(names):
                    if nm == str(_start_at) or str(_start_at) in nm:
                        start_idx = i
                        break
            if start_idx is not None:
                before = len(self._test_indices)
                self._test_indices = [i for i in self._test_indices if i >= start_idx]
                if self.rank == 0:
                    print(f"[Tester] DEBUG start_at={_start_at!r} -> starting at dataset "
                          f"index {start_idx}; dropped {before - len(self._test_indices)} "
                          f"earlier chunk(s) on rank 0 (per-rank). Unset test.start_at for a full run.")
            elif self.rank == 0:
                print(f"[Tester] start_at={_start_at!r} not found; running full split.")

        # Resume fast-path: drop chunks that already have a saved result so the
        # DataLoader never loads them (the main loop would skip them anyway, but
        # only after paying the load). Disjoint per-rank shards, so checking each
        # rank's own results is safe.
        if self.results_dir is not None and hasattr(dataset, "chunk_paths"):
            paths = dataset.chunk_paths
            kept = []
            for ci in self._test_indices:
                p = Path(paths[ci])
                name, area = p.name, p.parent.name
                out_path = (self.results_dir / area / f"{name}.npy"
                            if area else self.results_dir / f"{name}.npy")
                if not out_path.exists():
                    kept.append(ci)
            n_done = len(self._test_indices) - len(kept)
            if n_done:
                print(f"[Tester r{self.rank}] resume: {n_done} chunk(s) already done -> "
                      f"not loaded; {len(kept)} remaining.")
                self._resumed = True
            self._test_indices = kept

        loader = self._make_test_loader(dataset, self._test_indices, pin_memory=False)
        return loader, dataset

    def _build_test_dataset(self):
        """Build the test dataset. Uses cfg.test.transform when present via
        build_dataset; otherwise builds the dataset with no transform."""
        from datasets.builder import build_dataset, DATASETS

        _test_cfg = getattr(self._raw_cfg, "test", None)
        if _test_cfg is not None and hasattr(_test_cfg, "transform"):
            return build_dataset(self._raw_cfg, "test")

        dataset_cfg = OmegaConf.to_container(self._raw_cfg.dataset, resolve=True)
        dataset_cfg["data_root"] = self.cfg.data.data_root
        dataset_cfg["split"] = "test"
        dataset_cfg["transform"] = None
        return DATASETS.build(dataset_cfg)
