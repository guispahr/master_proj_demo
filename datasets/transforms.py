"""
Some functions are directly taken from 
https://github.com/Pointcept/Pointcept/blob/main/pointcept
"""

from collections.abc import Sequence, Mapping
import copy

import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy
import scipy.interpolate
import scipy.ndimage


from utils.registry import Registry
from datasets.utils import _pair

TRANSFORMS = Registry("transforms")

# When you sample points, only the keys will be indexed with the selected indices.
# So when custom keys in data dict, use Update with custom keys (e.g ground_truth)
def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "superpoint",
            "strength",
            "segment",
            "instance",
            "image_coord",
            "image_mask"
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            elif key == "index_valid_keys":
                data_dict_[key] = copy.copy(data_dict[key])
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

# Usefull to directly create data dictionary for PTV3
@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        keys = elements we keep in data_dict
        feat_keys = elements we concat to build features
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data

@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict

@TRANSFORMS.register_module()
class ToTensor(object):
    def __init__(self):
        self.image_to_tensor = ToTensorV2()
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            if "image" in data.keys():
                imgs = data["image"]
                # `len(imgs)` works for both a list[ndarray] and a stacked ndarray.
                # The Phase-3 contract returns variable-CAM_i per sample; produce
                # `(CAM_i, C, H, W)` (no leading batch dim) so the collate
                # concatenates along dim 0 into packed `(sum CAM_i, C, H, W)`.
                if len(imgs) > 0:
                    data["image"] = torch.stack(
                        [self.image_to_tensor(image=im)["image"] for im in imgs]
                    )
                else:
                    # zero-cam sentinel — never expected on GridNet but keeps the
                    # collate happy if a chunk happens to have no valid cameras.
                    data["image"] = torch.empty(0, 3, 1, 1)
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")
        
@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = (
                    np.random.rand(data_dict["coord"].shape[0]) * 1e-3,
                    np.array([]),
                )
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(
                        np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2),
                        1,
                    )
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "grid_coord" in data_dict.keys():
                        data_crop_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    if "displacement" in data_dict.keys():
                        data_crop_dict["displacement"] = data_dict["displacement"][
                            idx_crop
                        ]
                    if "strength" in data_dict.keys():
                        data_crop_dict["strength"] = data_dict["strength"][idx_crop]
                    if "image_coord" in data_dict.keys():
                        data_crop_dict["image_coord"] = data_dict["image_coord"][
                            idx_crop
                        ]
                    if "image_mask" in data_dict.keys():
                        data_crop_dict["image_mask"] = data_dict["image_mask"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(
                        1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"])
                    )
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(
                        np.concatenate((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "displacement" in data_dict.keys():
                data_dict["displacement"] = data_dict["displacement"][idx_crop]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx_crop]
            if "image_coord" in data_dict.keys():
                data_dict["image_coord"] = data_dict["image_coord"][idx_crop]
            if "image_mask" in data_dict.keys():
                data_dict["image_mask"] = data_dict["image_mask"][idx_crop]
        return data_dict
    
@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        # Use multiple grid sizes during training to improve robustness to point density / resolution.
        grid_sizes = None,# e.g [0.04, 0.05, 0.06]
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.grid_sizes = self.grid_sizes = None if grid_sizes is None else np.asarray(grid_sizes, dtype=float)
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
    
    def get_grid_size(self):
        if self.mode == "train" and self.grid_sizes is not None:
            return np.random.choice(self.grid_sizes)
        return self.grid_size
    
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        grid_size = self.get_grid_size()
        scaled_coord = data_dict["coord"] / np.array(grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                if "grid_coord" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                if "displacement" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    if "grid_coord" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_part["displacement"] = displacement[idx_part]
                    if "displacement" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

@TRANSFORMS.register_module() 
class SmartPointSampler:
    def __init__(self, max_points=30000, class_weights=None, ignore_index=-1, ignore_kept_weight = 0.05):
        self.max_points = max_points
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.ignore_kept_weight = ignore_kept_weight

        if class_weights is not None:
            self.class_weights = np.array(class_weights)

    def __call__(self, data_dict):
        coords = data_dict["coord"]
        labels = data_dict["segment"]

        N = coords.shape[0]
        if N <= self.max_points:
            return data_dict

        weights = np.ones(N)

        valid_mask = labels != self.ignore_index

        if self.class_weights is not None:
            weights[valid_mask] = self.class_weights[labels[valid_mask]]

        weights[~valid_mask] = self.ignore_kept_weight

        # --- NumPy sampling replacement ---
        weights = weights.astype(float)

        total = weights.sum()
        if total == 0:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / total

        idxs = np.random.choice(
            len(weights),
            size=self.max_points,
            replace=False,
            p=probs
        )

        return index_operator(data_dict, idxs)
        
@TRANSFORMS.register_module()
class RandomBlockCrop:
    """Training-only spatial crop: keep every point inside an axis-aligned XY
    window (full Z, full density).

    Reduces points-per-sample WITHOUT changing resolution/density

    Args:
        block_size : (size_x, size_y) window size in metres (scalar → square).
        mode       : "random" → window placed at a random position fully inside
                     the cloud's XY bbox; "center" → centred on the bbox.
        max_points : optional cap. If the chosen window still holds more than this
                     many points, the window is shrunk about its centre
                     (sqrt scaling) to roughly hit the budget — still full
                     density, just a smaller area. None = size-only.

    No-op when the cloud already fits the window (and the point budget). Subsets
    via ``index_operator``. Do NOT add it to val/test transforms — evaluation 
    must see full scenes.
    """

    def __init__(self, block_size=(40.0, 40.0), mode="random", max_points=None):
        if isinstance(block_size, (int, float)):
            block_size = (block_size, block_size)
        self.block_size = (float(block_size[0]), float(block_size[1]))
        assert mode in ("random", "center"), "mode must be 'random' or 'center'"
        self.mode = mode
        self.max_points = int(max_points) if max_points is not None else None

    def __call__(self, data_dict):
        coord = data_dict.get("coord")
        if coord is None or coord.shape[0] == 0:
            return data_dict

        xy = np.asarray(coord[:, :2], dtype=np.float64)
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        extent = hi - lo
        bx, by = self.block_size

        # Cloud already fits the window (and the budget) → nothing to do.
        if (extent[0] <= bx and extent[1] <= by
                and (self.max_points is None or coord.shape[0] <= self.max_points)):
            return data_dict

        block = np.array([min(bx, extent[0]), min(by, extent[1])], dtype=np.float64)
        if self.mode == "center":
            origin = lo + (extent - block) / 2.0
        else:  # random origin, window kept fully inside the bbox
            origin = lo + np.random.uniform(0.0, 1.0, size=2) * np.maximum(extent - block, 0.0)
        center = origin + block / 2.0

        def mask_for(b):
            half = b / 2.0
            return ((np.abs(xy[:, 0] - center[0]) <= half[0])
                    & (np.abs(xy[:, 1] - center[1]) <= half[1]))

        mask = mask_for(block)

        # Optional density-preserving budget: shrink the window (not the density)
        # if the chosen block is still over max_points.
        if self.max_points is not None:
            count = int(mask.sum())
            if count > self.max_points and count > 0:
                block = block * (self.max_points / count) ** 0.5
                mask = mask_for(block)

        idx = np.where(mask)[0]
        if idx.size == 0:  # degenerate window — leave the cloud unchanged
            return data_dict
        return index_operator(data_dict, idx)


# Coordinate transforms

@TRANSFORMS.register_module()
class RandomZRotation:
    # Matches Pointcept/ditr RandomRotate(axis="z"): the angle is given in units
    # of pi, so angle_range=[-1, 1] -> [-180 deg, +180 deg]; it is applied with
    # probability p; and the nuScenes/S3DIS configs rotate about the origin
    # (ditr passes center=[0, 0, 0], which after CenterShift is the scene center).
    def __init__(self, angle_range=(-1, 1), center_mode="origin", p=0.5):
        self.theta_min, self.theta_max = _pair(angle_range)
        self.center_mode = center_mode
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
        coord = data_dict["coord"].copy()
        if self.center_mode == "centroid":
            center = np.mean(coord, axis = 0)
        elif self.center_mode == "bounding_box":
            center = (np.min(coord, axis=0) + np.max(coord, axis=0))/2
        else:  # "origin" -> rotate about [0, 0, 0], matching ditr center=[0,0,0]
            center = 0
        theta = np.random.uniform(self.theta_min, self.theta_max) * np.pi
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array([
            [cos, -sin, 0],
            [sin,  cos, 0],
            [0,    0,   1]
        ], dtype=coord.dtype)
        data_dict["coord"] = (coord-center) @ rot.T + center
        
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"] @ rot.T

        return data_dict
    
# Same as Pointcept for compatibility with DITR
@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

@TRANSFORMS.register_module()
class RandomScale:
    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_min, self.scale_max = scale_range

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        scale = np.random.rand() * (self.scale_max - self.scale_min) + self.scale_min
        data_dict["coord"] = coord * scale
        return data_dict

@TRANSFORMS.register_module()  
class RandomJitter:
    def __init__(self, sigma=0.005, clip=0.02):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        noise = np.random.randn(*data_dict["coord"].shape) * self.sigma
        noise = np.clip(noise, -self.clip, self.clip)
        data_dict["coord"] = data_dict["coord"] + noise
        return data_dict

@TRANSFORMS.register_module()   
class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict:
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict:
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict
    
@TRANSFORMS.register_module()
class CenterShift:
    def __init__(self, apply_z=False, center_mode = "centroid"):
        self.apply_z = apply_z
        assert center_mode in ["centroid", "bounding_box"], "center_mode must be 'centroid' or 'bounding_box'"
        self.center_mode = center_mode

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord = data_dict["coord"].copy()
            if self.center_mode == "centroid":
                center_xy = np.mean(coord[:, :2], axis = 0)
            elif self.center_mode == "bounding_box":
                center_xy = (np.min(coord[:, :2], axis=0) + np.max(coord[:, :2], axis=0))/2
            else:
                raise NotImplementedError
            
            z_shift = np.min(coord[:,2]) if self.apply_z else 0

            data_dict["coord"] -= np.array([center_xy[0], center_xy[1], z_shift], dtype=coord.dtype)
        return data_dict


@TRANSFORMS.register_module()
class RandomShift:
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict
    
@TRANSFORMS.register_module()
class RandomDropout:
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros(n, dtype=bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict

# COLORS

@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 255
        return data_dict
    
@TRANSFORMS.register_module()
class RandomColorJitter:
    def __init__(self, brightness=0.05, contrast=0.05, p=0.95):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
        if data_dict["color"] is None or data_dict["color"].shape[1] < 3:
            return data_dict

        feat = data_dict["color"]
        rgb = feat[:, :3]

        # Brightness
        b = 1 + (np.random.rand() * 2 - 1) * self.brightness
        rgb = rgb * b

        # Contrast
        mean = rgb.mean(axis=0, keepdims=True)
        c = 1 + (np.random.rand() * 2 - 1) * self.contrast
        rgb = (rgb - mean) * c + mean

        feat[:, :3] = np.clip(rgb, 0, 1)
        data_dict["color"] = feat
        return data_dict
    
@TRANSFORMS.register_module()
class AutoContrast:
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
        feat = data_dict["color"]
        if feat is None or feat.shape[1] < 3:
            return data_dict

        rgb = feat[:, :3]
        lo = rgb.min(axis=0, keepdims=True)
        hi = rgb.max(axis=0, keepdims=True)
        scale = 255.0 / (hi - lo + 1e-6)
        contrast = (rgb - lo) * scale

        blend = np.random.rand() if self.blend_factor is None else self.blend_factor
        feat[:, :3] = (1 - blend) * rgb + blend * contrast
        data_dict["color"] = feat
        return data_dict

# Version of Pointcept's AutoContrast that computes contrast on each point's own RGB values rather than the global min/max.
@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict
    
@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict
    
@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict

@TRANSFORMS.register_module()
class ColorDrop:
    """Randomly drops RGB channels as proposed in https://arxiv.org/pdf/2206.04670"""
    def __init__(self, p=0.2):
        self.color_drop = p

    def __call__(self, data_dict):
        feat = data_dict["color"]
        if feat.shape[1] < 3:
            return data_dict
        feat = feat.copy()
        mask = (np.random.rand(feat.shape[0], 3) > self.color_drop).astype(feat.dtype)
        feat[:, :3] *= mask
        data_dict["color"] = feat
        return data_dict
    

@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict

    
# Images

@TRANSFORMS.register_module()
class CameraDropout:
    """Randomly mask out entire camera views (training regularisation).

    Drops cameras by setting image_mask to False for the selected slots.
    The raw image pixels are left intact so DINOv2 can still encode them —
    the mask alone controls which cameras contribute to fusion. This forces
    the model to be robust to partial camera coverage and prevents it from
    relying on a fixed camera layout as a shortcut.

    Args:
        p           : probability of dropping each individual camera slot.
        min_cameras : minimum number of cameras guaranteed to remain visible
                      (at least one slot with any visible point kept).
    """

    def __init__(self, p: float = 0.3, min_cameras: int = 1):
        self.p = p
        self.min_cameras = min_cameras

    def __call__(self, data_dict):
        if "image_mask" not in data_dict:
            return data_dict

        mask = data_dict["image_mask"]          # (N, CAM), numpy bool or torch bool
        n_cams = mask.shape[1]

        drop = np.array([random.random() < self.p for _ in range(n_cams)], dtype=bool)

        # Always keep at least min_cameras slots
        n_kept = int((~drop).sum())
        if n_kept < self.min_cameras:
            restore = np.where(drop)[0]
            np.random.shuffle(restore)
            drop[restore[: self.min_cameras - n_kept]] = False

        for i in np.where(drop)[0]:
            if isinstance(mask, np.ndarray):
                mask[:, i] = False
            else:
                mask[:, i] = False          # works for both numpy and torch

        data_dict["image_mask"] = mask
        return data_dict


def _num_cameras(data_dict) -> int:
    """Number of camera slots in a sample (0 when there are no images)."""
    if "image_mask" in data_dict and data_dict["image_mask"] is not None:
        return data_dict["image_mask"].shape[1]
    img = data_dict.get("image", None)
    if img is None:
        return 0
    return len(img) if isinstance(img, (list, tuple)) else img.shape[0]


def _subset_cameras(data_dict, idx):
    """Keep only the camera slots in ``idx`` (iterable of indices), consistently
    across ``image`` (list or (CAM,H,W,3) ndarray), ``image_coord`` (N,CAM,2)
    and ``image_mask`` (N,CAM). Operates in place and returns data_dict."""
    idx = list(idx)
    img = data_dict.get("image", None)
    if img is not None:
        if isinstance(img, (list, tuple)):
            data_dict["image"] = [img[i] for i in idx]
        else:
            data_dict["image"] = img[idx]
    if "image_coord" in data_dict and data_dict["image_coord"] is not None:
        data_dict["image_coord"] = data_dict["image_coord"][:, idx]
    if "image_mask" in data_dict and data_dict["image_mask"] is not None:
        data_dict["image_mask"] = data_dict["image_mask"][:, idx]
    return data_dict


@TRANSFORMS.register_module()
class RandomImageSelect:
    """Randomly keep a random NUMBER of camera views (training augmentation).

    Unlike :class:`CameraDropout` (which only masks slots but still feeds every
    image to the 2D backbone), this physically subsets ``image`` /
    ``image_coord`` / ``image_mask`` so the dropped cameras are never encoded —
    saving 2D-backbone compute & VRAM and forcing robustness to a variable
    number of views.

    Each call samples ``k`` uniformly in ``[min_images, max_images]`` (clamped to
    the number of available cameras) and keeps a random ``k``-subset of slots.

    Args:
        min_images : minimum number of cameras to keep (>= 1).
        max_images : maximum number to keep; ``None`` = all available cameras.
    """

    def __init__(self, min_images: int = 1, max_images: int | None = None):
        assert min_images >= 1, "min_images must be >= 1"
        self.min_images = min_images
        self.max_images = max_images

    def __call__(self, data_dict):
        if "image" not in data_dict:
            return data_dict
        n_cam = _num_cameras(data_dict)
        if n_cam <= 1:
            return data_dict
        hi = n_cam if self.max_images is None else min(self.max_images, n_cam)
        lo = max(1, min(self.min_images, hi))
        k = random.randint(lo, hi)
        if k >= n_cam:
            return data_dict
        idx = sorted(random.sample(range(n_cam), k))
        return _subset_cameras(data_dict, idx)


@TRANSFORMS.register_module()
class ClipImages:
    """Cap the number of camera views to at most ``max_images``.

    No-op when a sample already has ``<= max_images`` cameras ("if not already
    done"). When over the cap, keeps the first ``max_images`` slots
    (deterministic) or a random subset when ``random_select=True``. Use in
    train/val/test to bound per-sample 2D compute & VRAM regardless of how many
    cameras a dataset provides.

    Args:
        max_images    : hard cap on the number of camera views (>= 1).
        random_select : if True, keep a random ``max_images``-subset instead of
                        the first ``max_images``.
    """

    def __init__(self, max_images: int, random_select: bool = False):
        assert max_images >= 1, "max_images must be >= 1"
        self.max_images = max_images
        self.random_select = random_select

    def __call__(self, data_dict):
        if "image" not in data_dict:
            return data_dict
        n_cam = _num_cameras(data_dict)
        if n_cam <= self.max_images:
            return data_dict
        if self.random_select:
            idx = sorted(random.sample(range(n_cam), self.max_images))
        else:
            idx = list(range(self.max_images))
        return _subset_cameras(data_dict, idx)


class _ImageTransform(object):
    """
    Applies an albumentations transform to the image and encodes image_coord
    as bounding boxes to correctly transform them as well.
    """

    def __init__(self, transform):
        self.transform_keypoints = "keypoints" in transform.available_keys
        self.transform = A.Compose(
            [transform],
            keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        )

    def __call__(self, data_dict):
        if "image" in data_dict.keys():
            # Skip entirely when the chunk has no valid cameras — np.stack on an
            # empty list errors, and there's nothing to transform anyway.
            if len(data_dict["image"]) == 0:
                return data_dict
            if self.transform_keypoints:
                transformed = [
                    self.transform(
                        image=im,
                        keypoints=data_dict["image_coord"][:, i][
                            data_dict["image_mask"][:, i]
                        ],
                    )
                    for i, im in enumerate(data_dict["image"])
                ]
                data_dict["image"] = np.stack([t["image"] for t in transformed])
                for i, t in enumerate(transformed):
                    data_dict["image_coord"][:, i][data_dict["image_mask"][:, i]] = t[
                        "keypoints"
                    ]

                # update mask
                _, H, W, _ = data_dict["image"].shape
                data_dict["image_mask"] &= np.all(data_dict["image_coord"] >= 0, axis=2)
                data_dict["image_mask"] &= data_dict["image_coord"][..., 0] < W
                data_dict["image_mask"] &= data_dict["image_coord"][..., 1] < H
            else:
                data_dict["image"] = np.stack(
                    [self.transform(image=im)["image"] for im in data_dict["image"]]
                )
        return data_dict


@TRANSFORMS.register_module()
class ImageNormalize(_ImageTransform):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__(A.Normalize(mean=mean, std=std))


@TRANSFORMS.register_module()
class ImageColorJitter(_ImageTransform):
    def __init__(
        self,
        brightness: float | tuple[float, float] | None = None,
        contrast: float | tuple[float, float] | None = None,
        saturation: float | tuple[float, float] | None = None,
        hue: float | tuple[float, float] | None = None,
    ):
        super().__init__(
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )


@TRANSFORMS.register_module()
class ImageResize(_ImageTransform):
    def __init__(self, size: tuple[int, int]):
        super().__init__(A.Resize(height=size[0], width=size[1]))


@TRANSFORMS.register_module()
class ImageRandomHorizontalFlip(_ImageTransform):
    def __init__(self, p: float = 0.5):
        super().__init__(A.HorizontalFlip(p=p))


@TRANSFORMS.register_module()
class ImageCenterCrop(_ImageTransform):
    def __init__(self, size: tuple[int, int]):
        super().__init__(A.CenterCrop(height=size[0], width=size[1]))


@TRANSFORMS.register_module()
class ImageRandomCrop(_ImageTransform):
    def __init__(self, size: tuple[int, int]):
        super().__init__(A.RandomCrop(height=size[0], width=size[1]))


@TRANSFORMS.register_module()
class ImageRemove(object):
    def __init__(self, indices: list[int]):
        self.indices = indices

    def __call__(self, data_dict):
        if data_dict["image"].shape[0] == len(set(self.indices)):
            # keep at least one image, set mask to False
            data_dict["image"] = data_dict["image"][:1]
            data_dict["image_coord"] = data_dict["image_coord"][:, :1]
            data_dict["image_mask"] = data_dict["image_mask"][:, :1]
            data_dict["image_mask"][:] = False
        else:
            data_dict["image"] = np.delete(data_dict["image"], self.indices, axis=0)
            data_dict["image_coord"] = np.delete(
                data_dict["image_coord"], self.indices, axis=1
            )
            data_dict["image_mask"] = np.delete(
                data_dict["image_mask"], self.indices, axis=1
            )
        return data_dict