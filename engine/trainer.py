"""
Trainer for point clouds
"""
from __future__ import annotations
_DEBUG = False  # set True to print per-epoch batch counters


from abc import abstractmethod
import gc
import io
import random
from functools import partial
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf

from utils.config import Config
from utils.dist import (
    init_dist, get_rank, get_world_size, get_local_rank, set_seed, worker_init_fn,
)
from losses.loss_manager import LossManager
from datasets.builder import build_dataset
from datasets.utils import point_collate_fn
from metrics.metrics_manager import MetricManager


class ModelEMA:
    """Exponential moving average of model weights (params + buffers).

    Tracks the unwrapped module (pass model.module under DDP); the shadow is
    kept in float32. For validation the shadow is swapped into the live model
    (store -> copy_to -> restore) so model selection and best.pt use the
    smoothed weights while training continues from the raw ones. Floating
    params are averaged; integer buffers (e.g. BN num_batches_tracked) are
    copied verbatim.
    """

    def __init__(self, model, decay: float = 0.9998):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    @torch.no_grad()
    def copy_to(self, model):
        msd = model.state_dict()
        for k, v in msd.items():
            v.copy_(self.shadow[k].to(v.dtype))

    @torch.no_grad()
    def store(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        msd = model.state_dict()
        for k, v in msd.items():
            v.copy_(self._backup[k])
        self._backup = None

    def model_state_dict(self, model):
        """Shadow cast back to the model's dtypes, keyed like model.state_dict()."""
        return {k: self.shadow[k].to(v.dtype) for k, v in model.state_dict().items()}

    def load_shadow(self, shadow: dict):
        for k, v in shadow.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].dtype))


class BaseTrainer:
    """Task-agnostic training loop.

    Owns the generic plumbing: DDP, AMP, EMA, checkpointing, NaN recovery,
    early stopping and logging. Subclasses provide the model, data, loss,
    metrics and optimizer through the abstract methods and hooks below.
    """

    def __init__(self, cfg=None):
        self._raw_cfg = cfg  # keep original OmegaConf for to_container() calls
        self.cfg = Config(cfg)

        # Distributed setup - must come before device & dir setup
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

        # Seed before the model build. set_seed also seeds NumPy, which the
        # transforms rely on; the base seed is reused by the dataloader workers.
        _seed = getattr(self.cfg.training, "seed", None)
        if _seed is None:
            _seed = random.randint(0, 2**31 - 1)
        self.seed = int(_seed)
        set_seed(self.seed + self.rank)
        if self.rank == 0:
            print(f"[seed] base seed = {self.seed} (per-rank / per-worker offsets applied)")

        # Dirs - rank 0 creates, barrier so non-main ranks don't proceed until dirs exist
        self.save_dir = Path(self.cfg.outputs.save_dir)
        self.wdir = self.save_dir / "weights"
        if self.rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.wdir.mkdir(parents=True, exist_ok=True)
        if self.is_ddp:
            dist.barrier()

        self.last = self.wdir / "last.pt"
        self.best = self.wdir / "best.pt"
        self.csv = self.save_dir / "results.csv"

        # Save the fully-resolved config so every run is self-contained.
        cfg_path = self.save_dir / "config.yaml"
        _resuming = bool(getattr(getattr(cfg, "training", cfg), "resume", None))
        if self.rank == 0 and not _resuming:
            print("Fresh run, writing config")
            OmegaConf.save(
                OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
                cfg_path,
            )

        self.train_batch_size = self.cfg.training.batch_size
        self.val_batch_size = self.cfg.val.batch_size
        self.epochs = self.cfg.training.epochs

        self.start_epoch = 0
        self.epoch = 0
        self.global_step = 0

        # Model and dataset
        self.model = None
        self.train_loader = None
        self.val_loader = None

        # Optimization
        self.scheduler = None
        self.scaler = None
        self.optimizer = None

        self.amp = self.cfg.training.amp
        _dtype_str = getattr(self.cfg.training, "amp_dtype", "float16")
        self.amp_dtype = getattr(torch, _dtype_str)
        self.grad_accum = getattr(self.cfg.training, "grad_accum", 1)
        self.grad_clip = getattr(self.cfg.training, "grad_clip", 10.0)
        self.debug_nan = getattr(self.cfg.training, "debug_nan", False)

        self.save_period = self.cfg.training.save_period
        self.eval_every = self.cfg.training.eval_every

        # Metrics and loss tracking
        self.loss_manager = None
        self.metric_manager = None
        self.train_metrics = {}
        self.val_metrics = {}
        self.metrics = {}   # alias to train_metrics, kept for save_model
        self.tloss = None
        self.best_fitness = 0.0
        self.no_improve_epochs = 0
        self.patience = getattr(self.cfg.training, "early_stopping_patience", 0)
        self.warmup_epochs = 0  # set by _setup_scheduler in subclasses

        # Weight EMA, off by default. When enabled, validation and best.pt
        # use the EMA weights while training continues on the raw ones.
        self.use_ema = bool(getattr(self.cfg.training, "ema", False))
        self.ema_decay = float(getattr(self.cfg.training, "ema_decay", 0.9998))
        self.ema = None  # created in setup_train once the model is built

        self.loggers = []
        self.train_time_start = None

        self.setup_train()

    # Setup

    def setup_train(self):
        """Prepare model, optimizer, loss, metrics, dataloaders and loggers."""

        ckpt = self.setup_model()

        if self.model is None:
            raise RuntimeError("setup_model() must define self.model")

        self.model = self.model.to(self.device)

        # Freeze layers if specified (must run on raw model before DDP wrapping)
        freeze_list = getattr(self.cfg, "freeze", []) or []
        freeze_layer_names = [f"model.{x}." for x in freeze_list]
        for name, param in self.model.named_parameters():
            if any(x in name for x in freeze_layer_names):
                param.requires_grad = False

        # Optionally pool BatchNorm statistics across GPUs (matches ditr's sync_bn)
        # so BN stats do not depend on the per-GPU batch size. Must run on the raw
        # model before the DDP wrap. Enable with training.sync_bn: true.
        if self.is_ddp and getattr(self.cfg.training, "sync_bn", False):
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.rank == 0:
                print("[sync_bn] BatchNorm -> SyncBatchNorm (cross-GPU BN stats)")

        # DDP wrapping - after freeze, before optimizer
        if self.is_ddp:
            # Datasets with image-free chunks can leave some image params without a
            # gradient on a rank, which breaks find_unused_parameters=False. Enable
            # training.ddp_find_unused_parameters: true for such runs.
            find_unused = bool(getattr(self.cfg.training, "ddp_find_unused_parameters", False))
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=find_unused,
                # Match ditr: each rank keeps its own BN buffers (no per-step
                # broadcast of running stats from rank 0). Gradients are still synced.
                broadcast_buffers=False,
            )

        # GradScaler only helps float16; bfloat16 has no overflow issue so we disable it
        _scaler_enabled = self.amp and self.device.type == "cuda" and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=_scaler_enabled)

        # EMA tracks the unwrapped module (consistent state_dict keys under DDP).
        if self.use_ema:
            _ema_ref = self.model.module if self.is_ddp else self.model
            self.ema = ModelEMA(_ema_ref, decay=self.ema_decay)
            if self.rank == 0:
                print(f"[EMA] enabled (decay={self.ema_decay})")

        self.train_loader = self.get_dataloader(mode="train", batch_size=self.train_batch_size)
        self.val_loader = self.get_dataloader(mode="val", batch_size=self.val_batch_size)

        self.optimizer = self.build_optimizer(self.model)
        self._setup_scheduler()

        self.loss_manager = self.setup_loss()
        self.metric_manager = self.setup_metrics()
        self.loggers = self.setup_loggers()

        if ckpt:
            self._load_checkpoint_state(ckpt)

    def setup_loggers(self):
        """Build loggers from config, defaulting to ConsoleLogger + CSVLogger."""
        from loggers import LOGGERS
        from loggers.loggers import ConsoleLogger, CSVLogger

        loggers_cfg = getattr(self._raw_cfg, "loggers", None)
        if loggers_cfg is None:
            return [ConsoleLogger(), CSVLogger()]
        loggers_cfg = OmegaConf.to_container(loggers_cfg, resolve=True)
        return [LOGGERS.build(cfg) for cfg in loggers_cfg]

    # Training loop

    def train(self):

        self.train_time_start = time.time()

        if self.rank == 0:
            for logger in self.loggers:
                logger.on_train_start(self)

        for epoch in range(self.start_epoch, self.epochs):

            self.epoch = epoch
            self.val_metrics = {}   # reset - only populated if validate() runs this epoch

            # DistributedSampler needs the epoch so each rank gets a distinct shuffle
            if self.is_ddp and getattr(self, "_train_sampler", None) is not None:
                self._train_sampler.set_epoch(epoch)

            self._before_train()
            self._do_train()
            self._after_train()

            is_last_epoch = (epoch == self.epochs - 1)
            if self.val_loader and (epoch % self.eval_every == 0 or epoch == 0 or is_last_epoch):
                self.validate()

            self._after_epoch()

            # Early-stopping: rank 0 decides, broadcast to all ranks so they exit together
            if self.is_ddp:
                stop = torch.tensor(
                    int(self.patience > 0 and self.epoch >= self.warmup_epochs
                        and self.no_improve_epochs >= self.patience),
                    dtype=torch.int32, device=self.device,
                )
                dist.broadcast(stop, src=0)
                if stop.item():
                    if self.rank == 0:
                        print(f"Early stopping: no improvement for {self.patience} validation rounds.")
                    break
            else:
                if self.patience > 0 and self.epoch >= self.warmup_epochs and self.no_improve_epochs >= self.patience:
                    print(f"Early stopping: no improvement for {self.patience} validation rounds.")
                    break

        self._finish()

        if self.rank == 0:
            for logger in self.loggers:
                logger.on_train_end(self)

    def _do_train(self):

        self.model.train()
        self._on_train_epoch_start()
        total_loss = 0
        # Per-loss-component accumulator, filled when compute_loss sets
        # self.loss_items so each loss term can be reported separately.
        loss_components_sum: dict[str, float] = {}
        loss_components_count = 0

        if self.rank == 0:
            for logger in self.loggers:
                logger.on_train_epoch_start(self)

        if _DEBUG:
            _n_batches = _n_samples = _n_points = 0

        for i, batch in enumerate(self.train_loader):

            batch = self._move_to_device(batch)

            if _DEBUG:
                _n_batches += 1
                _off = batch.get("offset")
                if _off is not None:
                    _n_samples += int(_off.shape[0])
                    _n_points  += int(_off[-1].item())

            with torch.autocast(self.device.type, enabled=self.amp, dtype=self.amp_dtype):
                outputs = self.model(batch)
                loss = self.compute_loss(outputs, batch) / self.grad_accum

            if not torch.isfinite(loss):
                if self.debug_nan and not getattr(self, "_nan_diagnosed_this_epoch", False):
                    self._diagnose_nan(outputs, batch, loss)
                    self._nan_diagnosed_this_epoch = True
                self._handle_nan_recovery(self.epoch)
                continue

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum == 0 or (i + 1) == len(self.train_loader):
                self.optimizer_step()

            loss_val = loss.item() * self.grad_accum
            total_loss += loss_val
            self.global_step += 1

            items = getattr(self, "loss_items", None)
            if isinstance(items, dict):
                for name, val in items.items():
                    loss_components_sum[name] = loss_components_sum.get(name, 0.0) + float(val)
                loss_components_count += 1

            self.metric_manager.update(*self._get_metric_args(outputs, batch))

            if self.rank == 0:
                for logger in self.loggers:
                    logger.on_batch_end(self, i, loss_val)

        if _DEBUG and self.rank == 0:
            print(
                f"[DEBUG] epoch {self.epoch}: "
                f"{_n_batches} batches, {_n_samples} samples, {_n_points:,} points",
                flush=True,
            )

        self.train_metrics = {
            "loss": total_loss / len(self.train_loader),
            **self.metric_manager.compute(),
        }
        if loss_components_count > 0:
            for name, s in loss_components_sum.items():
                self.train_metrics[f"loss/{name}"] = s / loss_components_count
        # Snapshot the LRs so every logger sees them without touching the optimizer.
        for i, pg in enumerate(self.optimizer.param_groups):
            label = pg.get("_label", f"group_{i}")
            self.train_metrics[f"lr/{label}"] = float(pg["lr"])
        if self.scheduler is not None and hasattr(self.scheduler, "last_step"):
            self.train_metrics["sched/last_step"] = int(self.scheduler.last_step)
        self.metrics = self.train_metrics   # kept for save_model compatibility
        self.tloss = self.train_metrics["loss"]
        self.metric_manager.reset()

    def _after_epoch(self):
        if self.rank != 0:
            return

        for logger in self.loggers:
            logger.on_train_epoch_end(self, self.train_metrics)

        is_best = False
        if self.val_metrics:
            for logger in self.loggers:
                logger.on_val_end(self, self.val_metrics)
            fitness = self.fitness(self.val_metrics)
            if fitness is not None and self.epoch >= self.warmup_epochs:
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.no_improve_epochs = 0
                    is_best = True
                else:
                    self.no_improve_epochs += 1

        if is_best or self.epoch % self.save_period == 0:
            self.save_model(is_best=is_best)

    @torch.no_grad()
    def validate(self):

        self.model.eval()
        # Validate on the EMA weights, then restore the raw ones for training.
        _ema_model = (self.model.module if self.is_ddp else self.model) if self.ema is not None else None
        if _ema_model is not None:
            self.ema.store(_ema_model)
            self.ema.copy_to(_ema_model)
        total_val_loss = 0.0
        num_batches = 0
        # Per-loss-component accumulator, same idea as in _do_train.
        loss_components_sum: dict[str, float] = {}
        loss_components_count = 0
        self._on_val_start()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        _val_t0 = time.time()
        print(f"[validate] rank={self.rank} start epoch={self.epoch}", flush=True)

        if self.rank == 0:
            for logger in self.loggers:
                logger.on_val_epoch_start(self)

        for i, batch in enumerate(self.val_loader):
            batch = self._move_to_device(batch)
            ctx = self._before_val_forward(batch)
            with torch.autocast(self.device.type, enabled=self.amp, dtype=self.amp_dtype):
                outputs = self.model(batch)
                total_val_loss += self.compute_loss(outputs, batch).item()
            num_batches += 1
            items = getattr(self, "loss_items", None)
            if isinstance(items, dict):
                for name, val in items.items():
                    loss_components_sum[name] = loss_components_sum.get(name, 0.0) + float(val)
                loss_components_count += 1
            self._update_val_metrics(outputs, batch, ctx)

            if self.rank == 0:
                for logger in self.loggers:
                    logger.on_val_batch_end(self, i)

        if self.is_ddp:
            buf = torch.tensor([total_val_loss, float(num_batches)], device=self.device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            total_val_loss, num_batches = buf[0].item(), int(buf[1].item())

        self.val_metrics = {
            "loss": total_val_loss / max(num_batches, 1),
            **self.metric_manager.compute(),
        }
        if loss_components_count > 0:
            for name, s in loss_components_sum.items():
                self.val_metrics[f"loss/{name}"] = s / loss_components_count
        self.val_metrics.update(self._extra_val_metrics())
        self.metric_manager.reset()

        if _ema_model is not None:
            self.ema.restore(_ema_model)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[validate] rank={self.rank} done in {time.time()-_val_t0:.1f}s", flush=True)
        if self.is_ddp:
            dist.barrier()

    # Optimization

    def optimizer_step(self):
        """Perform a single optimizer step with gradient unscaling and clipping."""
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

        if self.ema is not None:
            self.ema.update(self.model.module if self.is_ddp else self.model)

    # Checkpointing

    def save_model(self, is_best: bool = False):
        if self.rank != 0:
            return
        model_to_save = self.model.module if self.is_ddp else self.model
        raw_sd = model_to_save.state_dict()
        ema_sd = self.ema.model_state_dict(model_to_save) if self.ema is not None else None

        def _blob(model_sd):
            buf = io.BytesIO()
            torch.save(
                {
                    "epoch": self.epoch,
                    "model": model_sd,
                    # Keep the EMA shadow so a resumed run continues averaging.
                    "ema": self.ema.shadow if self.ema is not None else None,
                    "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                    "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                    "scaler": self.scaler.state_dict() if self.scaler else None,
                    "metrics": self.metrics,
                    "date": datetime.now().isoformat(),
                    "wandb_run_id": getattr(self, "wandb_run_id", None),
                },
                buf,
            )
            return buf.getvalue()

        # last.pt carries the raw training weights (for resume); best.pt and the
        # periodic snapshot carry the EMA weights when EMA is on (for inference).
        last_blob = _blob(raw_sd)
        self.last.write_bytes(last_blob)
        infer_blob = _blob(ema_sd) if ema_sd is not None else last_blob
        if is_best:
            self.best.write_bytes(infer_blob)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(infer_blob)

    # Utilities

    def _move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x) for x in batch)
        return batch

    def _get_memory(self, fraction=False):
        memory, total = 0, 0
        if self.device.type == "cuda":
            memory = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(self.device).total_memory
        elif self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        if fraction and total > 0:
            return memory / total
        return memory / 2**30

    def _clear_memory(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.device.type == "mps":
            torch.mps.empty_cache()

    # Output-routing hooks

    def _get_logits_for_metrics(self, outputs):
        """Pull the logits tensor out of the model output."""
        return outputs

    def _get_metric_args(self, outputs, batch):
        """Return (predictions, context) forwarded to metric_manager.update().

        Override when the model output is not directly consumable by the
        metrics, e.g. a dict that wraps the logits.
        """
        return self._get_logits_for_metrics(outputs), batch

    # Abstract methods

    @abstractmethod
    def setup_model(self):
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, model):
        raise NotImplementedError

    @abstractmethod
    def setup_loss(self):
        raise NotImplementedError

    @abstractmethod
    def setup_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, outputs, batch):
        raise NotImplementedError

    @abstractmethod
    def get_dataloader(self, mode="train", batch_size=16):
        raise NotImplementedError

    # Optional hooks

    def _setup_scheduler(self):
        pass

    def _load_checkpoint_state(self, ckpt):
        pass

    def _on_train_epoch_start(self):
        """Called at the start of each training epoch, right after model.train()."""
        pass

    def _on_val_start(self):
        """Called once per validate() call, before the batch loop."""
        pass

    def _before_val_forward(self, batch):
        """Called on each val batch before the forward pass.

        Whatever this returns is handed back to _update_val_metrics as ctx.
        Subclasses use it to grab batch state the model may modify in place.
        """
        return None

    def _update_val_metrics(self, outputs, batch, ctx):
        """Feed one val batch to the metric manager; ctx comes from _before_val_forward."""
        self.metric_manager.update(*self._get_metric_args(outputs, batch))

    def _extra_val_metrics(self):
        """Extra entries merged into val_metrics after the loss components."""
        return {}

    def _diagnose_nan(self, outputs, batch, loss):
        """Print a one-shot report locating the NaN: inputs, outputs, losses, weights, scaler."""
        sep = "=" * 60
        lines = [
            sep,
            f"[NaN Diagnosis] epoch={self.epoch}  step={self.global_step}",
            sep,
        ]

        lines.append("-- Inputs --")
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    finite = torch.isfinite(v).all().item()
                    lines.append(
                        f"  {k:20s}: finite={finite}  shape={list(v.shape)}"
                        + (f"  min={v.min().item():.4f}  max={v.max().item():.4f}" if finite else "  *** CONTAINS NaN/Inf ***")
                    )

        lines.append("-- Outputs --")
        _logits = self._get_logits_for_metrics(outputs)
        if isinstance(_logits, torch.Tensor):
            finite = torch.isfinite(_logits).all().item()
            lines.append(
                f"  outputs: finite={finite}  shape={list(_logits.shape)}"
                + (f"  min={_logits.min().item():.4f}  max={_logits.max().item():.4f}" if finite else "  *** CONTAINS NaN/Inf ***")
            )

        lines.append("-- Loss components --")
        try:
            with torch.no_grad():
                total, breakdown = self.loss_manager(outputs)
            for name, val in breakdown.items():
                import math
                finite = math.isfinite(val)
                lines.append(
                    f"  {name:30s}: {val:.6f}"
                    + ("" if finite else "  *** NaN/Inf ***")
                )
        except Exception as e:
            lines.append(f"  could not compute losses - {e}")

        lines.append("-- Model weights --")
        nan_params = [(n, p) for n, p in self.model.named_parameters() if not torch.isfinite(p).all()]
        if nan_params:
            lines.append(f"  *** {len(nan_params)} parameter tensor(s) contain NaN/Inf: ***")
            for n, p in nan_params[:5]:
                lines.append(f"    {n}  shape={list(p.shape)}")
            if len(nan_params) > 5:
                lines.append(f"    ... and {len(nan_params) - 5} more")
        else:
            lines.append("  All weights finite - weights were NOT the source")

        lines.append("-- AMP scaler --")
        lines.append(f"  current scale: {self.scaler.get_scale():.1f}")

        lines.append(sep)
        print("\n".join(lines))

    def _handle_nan_recovery(self, epoch):
        if self.rank == 0:
            print(f"NaN detected at epoch {epoch}, restoring last checkpoint")
        self.optimizer.zero_grad(set_to_none=True)
        if self.last.exists():
            ckpt = torch.load(self.last, map_location=self.device, weights_only=False)
            model_ref = self.model.module if self.is_ddp else self.model
            model_ref.load_state_dict(ckpt["model"], strict=False)
            if self.rank == 0:
                print("Last model state loaded")
            if ckpt.get("optimizer") and self.optimizer:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                if self.rank == 0:
                    print("Last optimizer state loaded")
            if ckpt.get("scaler") and self.scaler:
                self.scaler.load_state_dict(ckpt["scaler"])
                if self.rank == 0:
                    print("Last scaler state loaded")
            if self.ema is not None and ckpt.get("ema"):
                self.ema.load_shadow(ckpt["ema"])
                if self.rank == 0:
                    print("Last EMA state loaded")

    def fitness(self, metrics: dict):
        """Scalar score used to pick the best checkpoint; None disables tracking."""
        return None

    def _before_train(self):
        self._nan_diagnosed_this_epoch = False

    def _after_train(self):
        pass

    def _finish(self):
        pass


# Concrete implementation: point cloud segmentation

class SegmentationTrainer(BaseTrainer):

    def setup_model(self):
        from models import MODELS
        model_cfg = OmegaConf.to_container(self._raw_cfg.model, resolve=True)
        self.model = MODELS.build(model_cfg)

        resume = getattr(self.cfg.training, "resume", None)
        if resume:
            ckpt = torch.load(resume, map_location="cpu", weights_only=False)
            return ckpt
        return None

    def build_optimizer(self, model):
        from utils.optimizer import build_optimizer as _build_optimizer
        raw_pd = getattr(self._raw_cfg, "param_dicts", None)
        param_dicts_cfg = OmegaConf.to_container(raw_pd, resolve=True) if raw_pd is not None else None
        sched_cfg = self.cfg.scheduler
        base_lr = getattr(sched_cfg, "peak_lr", None) or getattr(sched_cfg, "max_lr", None)
        return _build_optimizer(model, self.cfg.optim, base_lr, param_dicts_cfg)

    def _setup_scheduler(self):
        import math
        steps_per_epoch = len(self.train_loader)
        # The scheduler steps once per optimizer step (every grad_accum batches),
        # so all lengths are counted in optimizer steps.
        opt_steps_per_epoch = math.ceil(steps_per_epoch / self.grad_accum)
        total_steps = self.cfg.training.epochs * opt_steps_per_epoch
        sched_cfg = self.cfg.scheduler
        sched_type = getattr(sched_cfg, "type", "LinearWarmupCosine")

        if sched_type == "OneCycleLR":
            from scheduler import OneCycleLRScheduler
            pct_start = getattr(sched_cfg, "pct_start", 0.3)
            self.warmup_epochs = round(pct_start * self.cfg.training.epochs)
            self.scheduler = OneCycleLRScheduler(
                self.optimizer,
                max_lr=sched_cfg.max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy=getattr(sched_cfg, "anneal_strategy", "cos"),
                div_factor=getattr(sched_cfg, "div_factor", 25.0),
                final_div_factor=getattr(sched_cfg, "final_div_factor", 1e4),
                three_phase=getattr(sched_cfg, "three_phase", False),
            )
        else:  # LinearWarmupCosine (default)
            from scheduler import LinearWarmupCosineScheduler
            warmup_epoch = getattr(sched_cfg, "warmup_epoch", None)
            warmup_steps = opt_steps_per_epoch * warmup_epoch if warmup_epoch else int(0.1 * total_steps)
            self.warmup_epochs = warmup_epoch if warmup_epoch else round(warmup_steps / opt_steps_per_epoch)
            self.scheduler = LinearWarmupCosineScheduler(
                self.optimizer,
                start_lr=sched_cfg.start_lr,
                peak_lr=sched_cfg.peak_lr,
                end_lr=sched_cfg.end_lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )

    def setup_loss(self):
        loss_configs = OmegaConf.to_container(self._raw_cfg.losses, resolve=True)
        return LossManager(loss_configs)

    def setup_metrics(self):
        from metrics.metrics import (
            SegmentationMetrics, RangeBinnedSegMetrics, BoundarySegMetrics,
        )
        nc = self.cfg.data.num_classes
        ig = self.cfg.data.ignore_index
        metrics = [SegmentationMetrics(num_classes=nc, ignore_index=ig, device=self.device)]

        # Extra eval metrics, opt-in through an eval_metrics config block.
        em = getattr(self.cfg, "eval_metrics", None)
        rb = getattr(em, "range_binned", None) if em is not None else None
        if rb is not None and getattr(rb, "enabled", True):
            metrics.append(RangeBinnedSegMetrics(
                num_classes=nc, ignore_index=ig,
                bins=getattr(rb, "bins", [20.0, 40.0]),
                origin=getattr(rb, "origin", None),
                device=self.device,
            ))
        bd = getattr(em, "boundary", None) if em is not None else None
        if bd is not None and getattr(bd, "enabled", True):
            metrics.append(BoundarySegMetrics(
                num_classes=nc, ignore_index=ig,
                radius=getattr(bd, "radius", 0.05),
                device=self.device,
            ))
        return MetricManager(metrics)

    def _get_logits_for_metrics(self, outputs):
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        return outputs

    def _get_metric_args(self, outputs, batch):
        logits = self._get_logits_for_metrics(outputs)
        # When using Point-Segmentor, outputs already contains 'segment' so we
        # can use it as the context dict; fall back to the original batch otherwise.
        meta = outputs if isinstance(outputs, dict) else batch
        return logits, meta

    def _on_train_epoch_start(self):
        # Optional epoch hook on the backbone (used by the curriculum
        # image-modality dropout to interpolate p_img/p_cam over epochs).
        model_ref = self.model.module if self.is_ddp else self.model
        if hasattr(model_ref, "backbone") and hasattr(model_ref.backbone, "set_epoch"):
            model_ref.backbone.set_epoch(self.epoch)

    def _on_val_start(self):
        # Gate statistics (image-fusion gates) averaged over the val epoch.
        self._gate_accum = {}
        self._gate_count = 0

    def _before_val_forward(self, batch):
        """Decide whether this val batch can be scored at full resolution.

        Full-resolution eval needs origin_segment plus the inverse map stored
        by GridSample; multi-scene batches also need origin_offset, otherwise
        we warn once and score at voxel level. The voxel offsets are cloned
        here because the backbone pools offset in place during the forward.

        Args:
            batch: the val batch dict, already on the device.

        Returns:
            (eval_full_res, voxel_offset), with voxel_offset None when
            full-resolution eval is off.
        """
        eval_full_res = (
            isinstance(batch, dict)
            and "inverse" in batch
            and "origin_segment" in batch
        )
        if (
            eval_full_res
            and batch["offset"].numel() > 1
            and "origin_offset" not in batch
        ):
            if self.rank == 0 and not getattr(self, "_warned_no_origin_offset", False):
                print(
                    "[validate] 'inverse'/'origin_segment' present but no "
                    "'origin_offset' and val batch has >1 scene - falling back to "
                    "voxel-level mIoU. Add offset_keys_dict={offset: coord, "
                    "origin_offset: origin_segment} to the val Collect for "
                    "full-resolution eval (matches ditr)."
                )
                self._warned_no_origin_offset = True
            eval_full_res = False
        voxel_offset = batch["offset"].clone() if eval_full_res else None
        return eval_full_res, voxel_offset

    def _update_val_metrics(self, outputs, batch, ctx):
        """Score one val batch, at full resolution when ctx allows it.

        Also accumulates the backbone's gate statistics when it exposes
        get_gate_stats.

        Args:
            outputs: model output (dict with logits, or a plain tensor).
            batch: the val batch dict, after the forward.
            ctx: (eval_full_res, voxel_offset) from _before_val_forward.
        """
        eval_full_res, voxel_offset = ctx
        if eval_full_res:
            logits = self._get_logits_for_metrics(outputs)
            logits_full = self._expand_voxel_to_origin(
                logits, batch["inverse"], voxel_offset, batch.get("origin_offset")
            )
            self.metric_manager.update(logits_full, {"segment": batch["origin_segment"]})
        else:
            self.metric_manager.update(*self._get_metric_args(outputs, batch))

        model_ref = self.model.module if self.is_ddp else self.model
        backbone = getattr(model_ref, "backbone", None)
        if backbone is not None and hasattr(backbone, "get_gate_stats"):
            for k, v in backbone.get_gate_stats().items():
                self._gate_accum[k] = self._gate_accum.get(k, 0.0) + v
            self._gate_count += 1

    def _extra_val_metrics(self):
        if getattr(self, "_gate_count", 0) > 0:
            return {k: v / self._gate_count for k, v in self._gate_accum.items()}
        return {}

    def _expand_voxel_to_origin(self, logits, inverse, voxel_offset, origin_offset):
        """Map voxel-level logits back to original-point resolution.

        inverse gives, for each original point, the index of its voxel within
        its own scene; for multi-scene batches those local ids are shifted by
        the scene's voxel base so they index the concatenated logits. Same
        remap ditr does with knn_query, but exact on the grid.

        Args:
            logits: (N_vox, C) voxel-level logits.
            inverse: (N_origin,) per-original-point local voxel index.
            voxel_offset: (B,) cumulative voxel counts, captured before the forward.
            origin_offset: (B,) cumulative original-point counts, or None for a
                single-scene batch.

        Returns:
            (N_origin, C) logits expanded onto the original points.
        """
        inverse = inverse.long()
        if origin_offset is None:
            # Single scene per batch - inverse is already global.
            return logits[inverse]
        voxel_offset = voxel_offset.long()
        origin_offset = origin_offset.long()
        voxel_starts  = torch.cat([voxel_offset.new_zeros(1),  voxel_offset[:-1]])
        origin_starts = torch.cat([origin_offset.new_zeros(1), origin_offset[:-1]])
        origin_counts = origin_offset - origin_starts
        base_per_point = torch.repeat_interleave(voxel_starts, origin_counts)
        return logits[inverse + base_per_point]

    def compute_loss(self, outputs, batch):
        # Point-Segmentor: 'segment' lives in outputs - no explicit targets needed.
        # Base-Segmentor:  outputs is a plain tensor; pass targets from batch.
        targets = None if isinstance(outputs, dict) else batch.get("segment")
        loss, loss_dict = self.loss_manager(outputs, targets)
        self.loss_items = loss_dict
        return loss

    def fitness(self, metrics: dict):
        return metrics.get("mIoU", 0.0)

    def get_dataloader(self, mode="train", batch_size=16):
        dataset = build_dataset(self._raw_cfg, mode)
        num_workers = self.cfg.training.num_workers
        prefetch_factor = getattr(self.cfg.training, "prefetch_factor", 4) if num_workers > 0 else None

        if self.is_ddp:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=(mode == "train"),
                drop_last=(mode == "train"),
            )
            shuffle = False
        else:
            sampler = None
            shuffle = (mode == "train")

        if mode == "train":
            self._train_sampler = sampler  # stored so train() can call set_epoch each epoch

        # batch_size in config is the total batch size across all GPUs
        per_gpu_batch = batch_size // self.world_size
        if self.rank == 0 and batch_size % self.world_size != 0:
            print(
                f"[Warning] batch_size={batch_size} is not divisible by "
                f"world_size={self.world_size}; using {per_gpu_batch} per GPU."
            )

        mix_prob = getattr(self.cfg, "mix_prob", 0) if mode == "train" else 0
        collate_fn_ = partial(point_collate_fn, mix_prob=mix_prob)
        # Per-worker RNG seeding so NumPy augmentations differ across workers
        # (PyTorch seeds torch/random per worker but NOT numpy). Matches ditr.
        init_fn = (
            partial(worker_init_fn, num_workers=num_workers, rank=self.rank, seed=self.seed)
            if num_workers > 0 else None
        )
        return DataLoader(
            dataset, per_gpu_batch,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_,
            pin_memory=(mode == "train"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
            worker_init_fn=init_fn,
            # ditr drops the partial last train batch; keep val/test complete.
            drop_last=(mode == "train"),
        )

    def _load_checkpoint_state(self, ckpt):
        self.start_epoch = ckpt["epoch"] + 1
        model_ref = self.model.module if self.is_ddp else self.model
        missing, unexpected = model_ref.load_state_dict(ckpt["model"], strict=False)
        if self.rank == 0:
            if missing:
                print(f"[Checkpoint] {len(missing)} missing keys (new layers will be randomly init):")
                for k in missing[:10]:
                    print(f"  - {k}")
                if len(missing) > 10:
                    print(f"  ... and {len(missing) - 10} more")
            if unexpected:
                print(f"[Checkpoint] {len(unexpected)} unexpected keys (ignored):")
                for k in unexpected[:10]:
                    print(f"  - {k}")
                if len(unexpected) > 10:
                    print(f"  ... and {len(unexpected) - 10} more")
            if not missing and not unexpected:
                print("[Checkpoint] Model weights loaded successfully (exact match).")
        if ckpt.get("optimizer") and self.optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.rank == 0:
                print("[Checkpoint] Optimizer state loaded successfully.")
        if ckpt.get("scheduler") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])
            if self.rank == 0:
                print("[Checkpoint] Scheduler state loaded successfully.")
        if ckpt.get("scaler") and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler"])
            if self.rank == 0:
                print("[Checkpoint] Scaler state loaded successfully.")
        if self.ema is not None and ckpt.get("ema"):
            self.ema.load_shadow(ckpt["ema"])
            if self.rank == 0:
                print("[Checkpoint] EMA state loaded successfully.")
        self._resume_wandb_run_id = ckpt.get("wandb_run_id")
        if self.rank == 0:
            print(f"Resumed from epoch {ckpt['epoch']}")

    def _after_train(self):
        self._clear_memory()


class DistillationTrainer(SegmentationTrainer):
    """Encoder-distillation pretraining trainer.

    Reuses the segmentation trainer's plumbing (model build, optimizer,
    scheduler, dataloader, AMP, DDP, NaN recovery, resume) and only overrides
    what changes for label-free pretraining: distillation metrics, fitness as
    the negated val distill loss (lower is better under the existing
    fitness > best_fitness comparison), checkpoints without the teacher
    weights, and an encoder-only export at the end of training.
    """

    def setup_metrics(self):
        from metrics.metrics import DistillationMetrics

        model_ref = self.model.module if self.is_ddp else self.model
        if hasattr(model_ref, "distill_stages"):
            num_stages = len(model_ref.distill_stages)
        else:
            num_stages = len(self._raw_cfg.model.distill_stages)
        if self.rank == 0:
            print(f"[DistillationTrainer] tracking {num_stages} distill stages")
        return MetricManager([
            DistillationMetrics(num_stages=num_stages, device=self.device)
        ])

    def _get_metric_args(self, outputs, batch):
        return outputs, batch

    def _get_logits_for_metrics(self, outputs):
        if isinstance(outputs, dict) and "distill_loss" in outputs.keys():
            return outputs["distill_loss"].detach().unsqueeze(0)
        return outputs

    def fitness(self, metrics: dict):
        d = metrics.get("distill_loss", None)
        return -float(d) if d is not None else None

    def save_model(self, is_best: bool = False):
        """Same as parent.save_model, minus teacher.* state_dict entries.

        The teacher is loaded from HF on every fresh build, so excluding its
        params shrinks each checkpoint by ~hundreds of MB.
        """
        if self.rank != 0:
            return
        model_to_save = self.model.module if self.is_ddp else self.model
        trimmed_sd = {
            k: v for k, v in model_to_save.state_dict().items()
            if not k.startswith("teacher.")
        }
        buf = io.BytesIO()
        torch.save({
            "epoch": self.epoch,
            "model": trimmed_sd,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "metrics": self.metrics,
            "date": datetime.now().isoformat(),
            "wandb_run_id": getattr(self, "wandb_run_id", None),
        }, buf)
        blob = buf.getvalue()
        self.last.write_bytes(blob)
        if is_best:
            self.best.write_bytes(blob)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(blob)

    def _finish(self):
        """Export an encoder-only checkpoint for downstream transfer.

        Strips the student. prefix and keeps only embedding.* and enc.* keys,
        so the resulting file loads directly into a fresh LitePT-Ditr-v2
        backbone via the pretrained_backbone pattern used by Utonia.
        """
        super()._finish()
        if self.rank != 0:
            return
        model_ref = self.model.module if self.is_ddp else self.model
        full_sd = model_ref.state_dict()
        enc_sd = {
            k.replace("student.", "", 1): v
            for k, v in full_sd.items()
            if k.startswith(("student.embedding.", "student.enc."))
        }
        student_cfg = OmegaConf.to_container(self._raw_cfg.model.student, resolve=True)
        out_path = self.wdir / "encoder_only.pt"
        torch.save({"state_dict": enc_sd, "config": student_cfg}, out_path)
        print(f"[DistillationTrainer] Encoder-only checkpoint written to {out_path} "
              f"({len(enc_sd)} tensors)")

