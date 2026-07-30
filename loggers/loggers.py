import time
import torch


def _scalar_metrics(metrics: dict) -> dict:
    """Keep only scalar values — filters out per-class tensors."""
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _expand_metrics(metrics: dict) -> dict:
    """Expand per-class 1-D tensors into scalar entries per class. 2-D tensors (confusion matrix) are skipped."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif hasattr(v, "ndim") and v.ndim == 1:
            for i, val in enumerate(v):
                out[f"{k}_class_{i}"] = float(val)
        elif hasattr(v, "__len__") and not hasattr(v, "ndim"):
            for i, val in enumerate(v):
                out[f"{k}_class_{i}"] = float(val)
    return out


class BaseLogger:
    """Base interface for training loggers. Override the hooks you need."""

    def on_train_start(self, trainer):
        pass

    def on_train_epoch_start(self, trainer):
        pass

    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        pass

    def on_train_epoch_end(self, trainer, metrics: dict):
        pass

    def on_val_epoch_start(self, trainer):
        pass

    def on_val_batch_end(self, trainer, batch_idx: int):
        pass

    def on_val_end(self, trainer, metrics: dict):
        pass

    def on_train_end(self, trainer):
        pass


class ConsoleLogger(BaseLogger):
    """Prints epoch metrics to stdout."""

    def on_train_epoch_end(self, trainer, metrics: dict):
        self._print("train", trainer, metrics)

    def on_val_end(self, trainer, metrics: dict):
        self._print("val  ", trainer, metrics)

    def _print(self, phase, trainer, metrics):
        lr = trainer.optimizer.param_groups[0]["lr"]
        body = " | ".join(f"{k}: {v:.4f}" for k, v in _scalar_metrics(metrics).items())
        print(f"Epoch {trainer.epoch + 1}/{trainer.epochs} | {phase} | {body} | lr: {lr:.2e}")


class CSVLogger(BaseLogger):
    """Appends epoch metrics to a CSV file (one row per phase per epoch)."""

    def on_train_epoch_end(self, trainer, metrics: dict):
        self._write(trainer, "train", metrics)

    def on_val_end(self, trainer, metrics: dict):
        self._write(trainer, "val", metrics)

    def _write(self, trainer, phase, metrics):
        scalars = _expand_metrics(metrics)
        keys = ["epoch", "phase", "time"] + list(scalars.keys())
        t = time.time() - trainer.train_time_start
        csv_path = trainer.csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = "" if csv_path.exists() else ",".join(keys) + "\n"
        values = [str(trainer.epoch + 1), phase, f"{t:.2f}"] + [f"{v:.6g}" for v in scalars.values()]
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(header + ",".join(values) + "\n")
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            cm_path = csv_path.parent / f"confusion_matrix_{phase}_epoch{trainer.epoch + 1}.csv"
            n = cm.shape[0]
            header = ",".join([f"pred_{i}" for i in range(n)])
            rows = "\n".join(",".join(str(int(v)) for v in row) for row in cm.tolist())
            with open(cm_path, "w", encoding="utf-8") as f:
                f.write(header + "\n" + rows + "\n")


class WandbLogger(BaseLogger):
    """Logs metrics to Weights & Biases."""

    def __init__(self, project, name=None, dir=None, **kwargs):
        self.project = project
        self.name = name
        self.dir = dir
        self.kwargs = kwargs

    def on_train_start(self, trainer):
        import wandb
        from omegaconf import OmegaConf
        resume_id = getattr(trainer, "_resume_wandb_run_id", None)
        wandb.init(
            project=self.project,
            name=self.name,
            dir=self.dir or str(trainer.save_dir),
            config=OmegaConf.to_container(trainer._raw_cfg, resolve=True),
            id=resume_id,
            resume="must" if resume_id else None,
            **self.kwargs,
        )
        trainer.wandb_run_id = wandb.run.id

    def on_train_epoch_end(self, trainer, metrics: dict):
        self._log(trainer, "train", metrics)

    def on_val_end(self, trainer, metrics: dict):
        self._log(trainer, "val", metrics)

    def _log(self, trainer, phase, metrics):
        import wandb
        import matplotlib.pyplot as plt
        expanded = _expand_metrics(metrics)
        log_dict = {"epoch": trainer.epoch + 1, **{f"{phase}/{k}": v for k, v in expanded.items()}}
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            import numpy as np
            cm_np = cm.numpy().astype(float)
            row_sums = cm_np.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_np, row_sums, where=row_sums > 0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax)
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cm_norm[i, j] < 0.5 else "black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{phase} confusion matrix (row-normalized)")
            log_dict[f"{phase}/confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)
        wandb.log(log_dict)

    def on_train_end(self, trainer):
        import wandb
        wandb.finish()


class ProgressBarLogger(BaseLogger):
    """Per-batch tqdm progress bars for train and val loops."""

    def __init__(self):
        self._train_pbar = None
        self._val_pbar = None

    def on_train_epoch_start(self, trainer):
        from tqdm import tqdm
        self._train_pbar = tqdm(
            total=len(trainer.train_loader),
            desc=f"Epoch {trainer.epoch + 1}/{trainer.epochs} [train]",
            leave=True,
            dynamic_ncols=True,
        )

    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        if self._train_pbar is None:
            return
        postfix = {"loss": f"{loss:.4f}"}
        if torch.cuda.is_available():
            postfix["mem"] = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
        if batch_idx % 10 == 0:
            metrics = trainer.metric_manager.compute()
            if "mIoU" in metrics:
                postfix["mIoU"] = f"{metrics['mIoU']:.3f}"
            if "OA" in metrics:
                postfix["OA"] = f"{metrics['OA']:.3f}"
        self._train_pbar.update(1)
        self._train_pbar.set_postfix(postfix)

    def on_train_epoch_end(self, trainer, metrics: dict):
        if self._train_pbar:
            self._train_pbar.close()
            self._train_pbar = None

    def on_val_epoch_start(self, trainer):
        from tqdm import tqdm
        self._val_pbar = tqdm(
            total=len(trainer.val_loader),
            desc=f"Epoch {trainer.epoch + 1}/{trainer.epochs} [val]  ",
            leave=False,
            dynamic_ncols=True,
        )

    def on_val_batch_end(self, trainer, batch_idx: int):
        if self._val_pbar:
            if batch_idx % 10 == 0:
                metrics = trainer.metric_manager.compute()
                postfix = {}
                if "mIoU" in metrics:
                    postfix["mIoU"] = f"{metrics['mIoU']:.3f}"
                if "OA" in metrics:
                    postfix["OA"] = f"{metrics['OA']:.3f}"
                if postfix:
                    self._val_pbar.set_postfix(**postfix)
            self._val_pbar.update(1)

    def on_val_end(self, trainer, metrics: dict):
        if self._val_pbar:
            self._val_pbar.close()
            self._val_pbar = None
