import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int):
    """Seed Python, NumPy and Torch RNGs (mirrors ditr/Pointcept's set_seed).

    Crucially seeds NumPy too: the data transforms use np.random heavily, and
    PyTorch does NOT auto-seed NumPy inside DataLoader workers — so without this
    every worker forks the same NumPy state and draws an identical augmentation
    stream, collapsing augmentation diversity.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    """Give every (rank, worker) a distinct, deterministic RNG stream.

    worker_seed = num_workers * rank + worker_id + base_seed  (same formula as ditr).
    """
    set_seed(num_workers * rank + worker_id + seed)


def init_dist():
    """Initialise the NCCL process group when running under torchrun.
    No-op when LOCAL_RANK is not set (single-GPU / plain python run).

    The NCCL watchdog default is 10 min, which is the *waiting slack* a fast rank
    will tolerate at a collective before aborting. During zone-aggregated testing
    whole zones are sharded per-rank ([tester.py] _test_zone_aggregated), so one
    rank can legitimately run much longer than the others before the metric-sync
    all-reduce; the fast ranks must wait that long without tripping the watchdog.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return
    torch.cuda.set_device(local_rank)
    timeout_min = int(os.environ.get("NCCL_TIMEOUT_MIN", "120"))
    dist.init_process_group(
        backend="nccl", timeout=timedelta(minutes=timeout_min)
    )


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))
