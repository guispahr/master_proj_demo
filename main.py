import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from omegaconf import OmegaConf as OC
from torch.utils.data import DataLoader

from datasets import *
from engine.trainer import SegmentationTrainer, DistillationTrainer
from engine.tester import SemSegTester

# Map trainer.type (YAML field) -> trainer class.
# Default ("segmentation") is used when the field is absent
TRAINER_CLS = {
    "segmentation": SegmentationTrainer,
    "distillation": DistillationTrainer,
}
from utils.profiling import (profile_data_loading, 
                             profile_model, 
                             profile_model_training, 
                             get_model_size, 
                             autobatch)

from models import MODELS

# FOR single GPU use : python main.py --config-file ...
# For multi-gpu use: torchrun --nproc_per_node=4 main.py --config-file ...
# Multi-GPU also works for --test-only: chunks (and whole zones, for the
# zone-aggregated path) are sharded across ranks (disjoint -> no output-file
# collisions) and the metric is reduced across ranks before reporting on rank 0.
def get_parser():
    parser = argparse.ArgumentParser("Point Cloud Segmentation")
    parser.add_argument("--config-file", default="configs/PTV3_segmentation.yaml", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--test-only", action="store_true", help="perform test only")
    parser.add_argument("--profiling", action="store_true", help="run profiling instead of training")
    return parser.parse_args()


def main():
    args = get_parser()

    def load_with_base(path):
        """Recursively resolve _base_ chains. Innermost base loads first;
        each layer's keys override the layer beneath. Lets ablation configs
        inherit from another config that itself inherits from base.yaml.
        """
        cfg = OC.load(path)
        base = cfg.get("_base_", None)
        if base:
            base_abs = Path(path).parent / base
            cfg = OC.merge(load_with_base(base_abs), cfg)
        return cfg

    cfg = load_with_base(args.config_file)
    OC.resolve(cfg)

    if args.profiling:
        from functools import partial
        from datasets.utils import collate_fn, point_collate_fn
        model_cfg = OC.to_container(cfg.model, resolve=True)
        model = MODELS.build(model_cfg)
        model.to(cfg.training.device)
        get_model_size(model)

        train_set = build_dataset(cfg, "train")
        val_set = build_dataset(cfg, "val")
        train_loader = DataLoader(train_set, cfg.training.batch_size, shuffle=True, persistent_workers=True,
                                  num_workers=cfg.training.num_workers, collate_fn=collate_fn, pin_memory=True)
        
        # from torch.amp import autocast
        # for batch in train_loader:
        #     N = batch["coord"].shape[0]
        #     if N >200000:
        #         batch = {k: v.cuda() for k, v in batch.items()}
        #         print(N)
        #         break
        # with autocast(device_type=cfg.training.device, enabled=True):
            
        #     outputs = model(batch)
        # # outputs = model(batch)
        # print(torch.cuda.memory_allocated()/1e9)
        # quit()
        
        print("Profiling data loading...")
        profile_data_loading(train_loader, cfg.training.device, 5)
        print("Profiling model forward pass...")
        profile_model(model, train_loader, cfg.training.device)
        print("Profiling model training step...")
        profile_model_training(model, 
                               train_loader, 
                               cfg.training.device,
                               cfg.training.amp, 
                               cfg.data.ignore_index,
                            )
        
        print("(AUTOBATCH) Defining the best batch size...")
        _amp_dtype = getattr(torch, cfg.training.amp_dtype)
        mix_prob = getattr(cfg, "mix_prob", 0)
        batch_size = autobatch(model,
                               train_set,
                               dataloader=DataLoader,
                               fraction=0.7,
                               amp=cfg.training.amp,
                               amp_dtype=_amp_dtype,
                               default_batch_size=1,
                               ignore_index= cfg.data.ignore_index,
                               collate_fn=partial(point_collate_fn, mix_prob=mix_prob),
                               loss_fn = None)
        print(f"Batch size adviced: {batch_size}")
        return

    # Auto-resume: if this run's output dir already has a checkpoint, pick up where it left off.
    # This handles preemption on low-priority clusters without any manual config change.
    _save_dir = Path(OC.select(cfg, "outputs.save_dir"))
    _last_ckpt = _save_dir / "weights" / "last.pt"
    _already_resuming = bool(OC.select(cfg, "training.resume", default=None))
    if _last_ckpt.exists() and not _already_resuming:
        print(f"[Auto-resume] Checkpoint found at {_last_ckpt} - resuming training.")
        OC.update(cfg, "training.resume", str(_last_ckpt), merge=True)

    trainer_type = OC.select(cfg, "trainer.type", default="segmentation")
    if trainer_type not in TRAINER_CLS:
        raise ValueError(
            f"Unknown trainer.type={trainer_type!r}; expected one of {list(TRAINER_CLS)}"
        )
    TrainerCls = TRAINER_CLS[trainer_type]

    if args.eval_only:
        trainer = TrainerCls(cfg)
        trainer.validate()
    elif args.test_only:
        # For test-only runs, load model/data/training config from the saved
        # config.yaml written by the trainer at the start of the run.  This
        # guarantees the model architecture matches the checkpoint even if the
        # current YAML has drifted.  Only the test block (checkpoint path,
        # split, transforms, TTA settings) is taken from the user's config.
        #TODO check if it works properly
        _saved_cfg_path = _save_dir / "config.yaml"
        if _saved_cfg_path.exists():
            saved_cfg = OC.load(_saved_cfg_path)
            test_block = OC.select(cfg, "test", default=OC.create({}))
            cfg = OC.merge(saved_cfg, OC.create({"test": OC.to_container(test_block)}))
            print(f"[Test] Model config loaded from {_saved_cfg_path}")
        else:
            print(f"[Test] No saved config found at {_saved_cfg_path} - using current config")
        tester = SemSegTester(cfg)
        tester.test()

    else:
        trainer = TrainerCls(cfg)
        trainer.train()


if __name__ == "__main__":
    main()
