"""
Pointcept-style base datasets.

`DefaultDataset` mirrors `pointcept/datasets/defaults.py`:
- generic `get_data_list` (glob over `data_root/split/*`),
- generic `get_data` (load every `.npy` whose stem is in `VALID_ASSETS`,
  normalise dtypes, default-fill missing `segment`/`instance` with -1),
- train/test dispatch via `__getitem__` and TTA fragment-list assembly in
  `prepare_test_data`.

"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.builder import DATASETS, build_transforms #, TRANSFORMS, Compose


@DATASETS.register_module()
class DefaultDataset(Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
    ]

    def __init__(
        self,
        split="train",
        data_root: str = "data/dataset",
        transform=None,
        test_mode: bool = False,
        test_cfg: dict | None = None,
        cache: bool = False,
        ignore_index: int = -1,
        loop: int = 1,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform               # already a Compose from build_transforms
        self.cache = cache                       # accepted for Pointcept API parity (no-op)
        self.ignore_index = ignore_index
        self.loop = 1 if test_mode else loop
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            assert test_cfg is not None, "test_mode requires test_cfg"
            self.test_voxelize = build_transforms({"GridSample": test_cfg["voxelize"]})
            crop = test_cfg.get("crop")
            self.test_crop = build_transforms(crop) if crop else None
            self.post_transform = build_transforms(test_cfg.get("post_transform", {}))
            self.aug_transform = [build_transforms(a) for a in test_cfg.get("aug_transform", [])]
        else:
            self.test_voxelize = None
            self.test_crop = None
            self.post_transform = None
            self.aug_transform = []

        self.data_list = self.get_data_list()
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"{self.__class__.__name__}: {len(self.data_list)} x {self.loop} samples in split '{self.split}'")

    # Hook points subclasses typically override

    def get_data_list(self):
        """Default: glob `data_root/split/*`. `split` may be a str or sequence."""
        if isinstance(self.split, str):
            return sorted(glob.glob(os.path.join(self.data_root, self.split, "*")))
        if isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += sorted(glob.glob(os.path.join(self.data_root, split, "*")))
            return data_list
        raise NotImplementedError(f"Unsupported split type: {type(self.split)}")

    def get_data(self, idx):
        """Default loader: read every `.npy` whose stem is in VALID_ASSETS.

        Normalises dtypes (coord/color/normal → float32; segment/instance → int32
        reshaped to (N,)). Missing `segment`/`instance` are filled with -1 so
        downstream code can index unconditionally.
        """
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        data_dict: dict = {}
        for asset in os.listdir(data_path):
            if not asset.endswith(".npy"):
                continue
            stem = asset[:-4]
            if stem not in self.VALID_ASSETS:
                continue
            data_dict[stem] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict:
            data_dict["coord"] = data_dict["coord"].astype(np.float32)
        if "color" in data_dict:
            data_dict["color"] = data_dict["color"].astype(np.float32)
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict:
            data_dict["segment"] = data_dict["segment"].reshape(-1).astype(np.int32)
        elif "coord" in data_dict:
            data_dict["segment"] = np.full(data_dict["coord"].shape[0], -1, dtype=np.int32)

        if "instance" in data_dict:
            data_dict["instance"] = data_dict["instance"].reshape(-1).astype(np.int32)
        elif "coord" in data_dict:
            data_dict["instance"] = np.full(data_dict["coord"].shape[0], -1, dtype=np.int32)

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    # Train / test data preparation (Pointcept-aligned)

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        result: dict = {"n_orig": int(data_dict["coord"].shape[0])}
        if "name" in data_dict:
            result["name"] = data_dict.pop("name")
        if "segment" in data_dict:
            result["segment"] = data_dict.pop("segment")
        if "inverse" in data_dict:
            result["inverse"] = data_dict.pop("inverse")
        if "origin_segment" in data_dict:
            result["origin_segment"] = data_dict.pop("origin_segment")

        # Image pixel data is the same for every fragment: spatial augmentations
        # (scale, flip) only move 3D coords and never change pixel values or
        # camera projections (image_coord stays in 2-D pixel space).
        # Convert to a (CAM, C, H, W) tensor once and store alongside the
        # fragment list; the tester reattaches it per-fragment at forward time.
        # This keeps the fragment list free of large pixel tensors, cutting
        # peak RAM by ~(n_frags - 1) × image_bytes.
        if "image" in data_dict:
            imgs = data_dict.pop("image")
            if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
                # (CAM, H, W, C) float32 after ImageNormalize stacked them
                result["shared_image"] = torch.from_numpy(
                    np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))
                ).float()
            elif isinstance(imgs, (list, tuple)) and len(imgs) > 0:
                stacked = np.stack(imgs)  # (CAM, H, W, C)
                result["shared_image"] = torch.from_numpy(
                    np.ascontiguousarray(stacked.transpose(0, 3, 1, 2))
                ).float()
            else:
                result["shared_image"] = torch.empty(0, 3, 1, 1)

        fragment_list = []
        for aug in self.aug_transform:
            aug_data = aug(deepcopy(data_dict))
            if self.test_voxelize is not None:
                parts = self.test_voxelize(aug_data)
            else:
                aug_data["index"] = np.arange(aug_data["coord"].shape[0])
                parts = [aug_data]
            if self.test_crop is not None:
                parts = [p for part in parts for p in self.test_crop(part)]
            fragment_list += parts

        fragment_list = [self.post_transform(f) for f in fragment_list]
        result["fragment_list"] = fragment_list
        return result

    def __getitem__(self, idx):
        return self.prepare_test_data(idx) if self.test_mode else self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ConcatDataset(Dataset):
    """Concatenate several `DefaultDataset` subclasses (Pointcept-aligned).

    Each entry of `datasets` is a config dict consumable by `DATASETS.build`.
    """

    def __init__(self, datasets: list[dict], loop: int = 1):
        super().__init__()
        self.datasets = [DATASETS.build(d) for d in datasets]
        self.loop = loop
        self.data_list = self.get_data_list()
        print(f"ConcatDataset: {len(self.data_list)} x {self.loop} samples across {len(self.datasets)} datasets")

    def get_data_list(self):
        data_list = []
        for i, ds in enumerate(self.datasets):
            data_list.extend(
                zip(
                    np.ones(len(ds), dtype=np.int32) * i,
                    np.arange(len(ds), dtype=np.int32),
                )
            )
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[int(dataset_idx)][int(data_idx)]

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[int(dataset_idx)].get_data_name(int(data_idx))

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
