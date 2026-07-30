"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os

import cv2
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module("S3DIS")
class S3DISDataset(DefaultDataset):
    def __init__(
        self,
        with_images=False,
        image_root="data/s3dis_images",
        **kwargs,
    ):
        self.with_images = with_images
        self.image_root = image_root
        super().__init__(**kwargs)

    def get_data_list(self):
        # Each Area dir contains room subdirs plus raw `Area_*_alignmentAngle.txt`
        # files; keep only directories so os.listdir() in get_data never hits a file.
        splits = [self.split] if isinstance(self.split, str) else self.split
        data_list = []
        for split in splits:
            for path in sorted(glob.glob(os.path.join(self.data_root, split, "*"))):
                if os.path.isdir(path):
                    data_list.append(path)
        return data_list

    def get_data_name(self, idx):
        remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(remain)
        return f"{area_name}-{room_name}"

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if self.with_images:
            H, W = 1080, 1080
            n_point = data_dict["coord"].shape[0]
            num_fill = 0
            images = []
            image_coords = []
            image_masks = []
            area, room = name.split("-")
            image_path = f"{self.image_root}/{area}/{room}/color/"
            visibility_path = f"{self.image_root}/{area}/{room}/visibility/"
            all_masks = glob.glob(os.path.join(visibility_path, "*_mask.npy"))
            if len(all_masks) <= self.with_images:
                selected_indices = np.arange(len(all_masks))
                num_fill = self.with_images - len(all_masks)
            elif self.split != "Area_5":
                selected_indices = np.random.choice(
                    len(all_masks), self.with_images, replace=False
                )
            else:
                step = len(all_masks) // self.with_images
                selected_indices = [i * step for i in range(self.with_images)]
            selected_masks = [all_masks[i] for i in selected_indices]
            for mask_path in selected_masks:
                visible_idx = np.load(mask_path).astype(np.int64)
                image_mask = np.zeros((n_point,), dtype=bool)
                image_mask[visible_idx] = True
                image_masks.append(image_mask)

                img_name = os.path.basename(mask_path).replace("_mask.npy", ".png")
                img_path = os.path.join(image_path, img_name)
                # images.append(imageio.imread(img_path))
                # replace with cv2 to avoid potential issues with imageio TODO check this
                images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)) 

                points = np.zeros((data_dict["coord"].shape[0], 2), dtype=np.float32)
                points_name = os.path.basename(mask_path).replace(
                    "_mask.npy", "_points.npy"
                )
                points_file = os.path.join(visibility_path, points_name)
                points[image_masks[-1]] = np.load(points_file)
                image_coords.append(points)
            for _ in range(num_fill):
                images.append(np.zeros((H, W, 3), dtype=np.uint8))
                image_coords.append(np.zeros((n_point, 2), dtype=np.float32))
                image_masks.append(np.zeros((n_point), dtype=bool))

            data_dict["image"] = np.stack(images, axis=0)  # (CAM, H, W, 3)
            data_dict["image_coord"] = np.stack(image_coords, axis=1)  # (N, CAM, 2)
            data_dict["image_mask"] = np.stack(image_masks, axis=1)  # (N, CAM)

        return data_dict