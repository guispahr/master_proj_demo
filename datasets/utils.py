"""
Dataset utils — collate functions and small helpers.

Mirrors `pointcept/datasets/utils.py` with one extension: when Mix3D fires on
an image-fusion batch (`image` / `image_coord` / `image_mask` present), the
camera dimension of each mixed pair is widened to fit both scenes' cameras
side by side (per pair: CAM_A + CAM_B columns, then the batch-wide max of
that across pairs) so per-scene image_coord/mask slots don't collide after
the offset merge. This equals 2*CAM only when every sample has the same
camera count. Pointcept's standard models don't read images, so they don't
need this.

"""

import random
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def _pair(value):
    """Coerce scalar or 2-tuple/list into a (float, float) tuple."""
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"Expected int/float or tuple of length 2. Got {type(value)}")


def sanitize_points(data_dict, max_half_extent=(30.0, 30.0, 150.0)):
    """Drop non-finite points and clip far outliers, in-place, at data-load time.

    A stray/corrupt point hundreds of metres from the chunk body inflates the
    voxel grid: ``sparse_shape ≈ max(grid_coord)+pad`` can then overflow spconv's
    int32 flat index → a *silent CUDA "illegal memory access"* deep in the model
    (see ``models/utils/structure.py:sparsify``). Removing such points at load
    prevents that.

    ``max_half_extent`` (metres) bounds each point to ``median ± max_half_extent``
    **per axis** — accepts a scalar or an ``(x, y, z)`` tuple. GridNet/Helimap
    chunks are tiled ~20 m in X/Y but the **Z (height) axis is not tiled**: pylons,
    towers, buildings, tall vegetation and terrain legitimately sit tens of metres
    above the ground-dominated median. So the bound is tight in X/Y (catches stray
    points beyond the tile) and generous in Z (keeps real tall structure). Once the
    far garbage point is removed the remaining (legitimate, small-extent) chunk no
    longer overflows; the sparsify int32 guard backstops any residual case.

    All per-point arrays in ``data_dict`` (those whose leading dim equals the point
    count) are filtered with the same mask, keeping coord/color/segment/
    image_coord/image_mask aligned.
    """
    coord = data_dict.get("coord")
    if coord is None:
        return data_dict
    coord = np.asarray(coord)
    if coord.ndim != 2 or coord.shape[0] == 0:
        return data_dict
    n = coord.shape[0]

    bound = np.asarray(max_half_extent, dtype=np.float64)  # scalar or (3,), per-axis
    keep = np.isfinite(coord).all(axis=1)
    valid = coord[keep]
    if valid.shape[0]:
        center = np.median(valid, axis=0)
        keep &= (np.abs(coord - center) <= bound).all(axis=1)

    if keep.all():
        return data_dict  # clean chunk → no-op

    n_drop = int((~keep).sum())
    warnings.warn(
        f"sanitize_points: dropped {n_drop}/{n} non-finite/outlier point(s) "
        f"(chunk='{data_dict.get('name', '?')}')."
    )
    for k, v in list(data_dict.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            data_dict[k] = v[keep]
    return data_dict


def to_float_rgb(rgb):
    rgb = rgb.float()
    if rgb.max() > 1:
        rgb = rgb / 255
    return rgb.clamp(min=0, max=1)


def get_las_attribute(las, attr):
    """Safely get LAS attribute. Returns None if not present."""
    if hasattr(las, attr):
        return getattr(las, attr)
    elif attr in las.point_format.extra_dimension_names:
        return las[attr]
    return None

def collate_fn(batch):
    """
    Generic recursive collate for point-cloud dicts (Pointcept-style).

    - Concatenates tensor lists along dim 0.
    - For dict batches, any key whose name contains "offset" is cumsum'd after
      concat (so per-sample point counts become absolute scene boundaries).
    - Sequences (lists/tuples of tensors per sample) are zipped and recursively
      collated; an extra per-sample size is appended so the collated sequence
      ends with cumulative offsets — matches the Pointcept convention.
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))

    if isinstance(batch[0], str):
        return list(batch)

    if isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch

    if isinstance(batch[0], Mapping):
        out = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in out:
            if "offset" in key:
                out[key] = torch.cumsum(out[key], dim=0)
        return out

    return default_collate(batch)


def point_collate_fn(batch, mix_prob: float = 0):
    """
    Point-cloud collate with optional Mix3D (https://arxiv.org/pdf/2110.02210)
    and variable-cameras-per-sample image packing.

    Image contract (Phase 3):
        Each sample carries `image: (CAM_i, C, H, W)` and `image_coord` /
        `image_mask` shaped `(N_i, CAM_i, …)`. `CAM_i` may differ between
        samples (and may be 0). This collate:
          - pads `image_coord` / `image_mask` to `max(CAM_i)` along the cam
            axis (CPU-side, cheap),
          - concatenates `image` along dim 0 → packed `(sum CAM_i, C, H, W)`
            — exactly the form the image-fusion models expect,
          - writes `cam_offset = cumsum(CAM_i)` so the model can split per
            scene.

    On Mix3D activation:
        - `scene_offset` is written (pre-mix per-scene boundaries) so losses
          can still distinguish individual scenes inside merged pairs.
        - `offset` and `cam_offset` are pair-merged (0↔1, 2↔3, …).
        - `image_coord` / `image_mask` are compacted per mega-scene: scene A's
          coords go in cols `[0:CAM_A]` and scene B's go in `[CAM_A:CAM_A+CAM_B]`.
          The `image` tensor is already in scene order, so it doesn't move.

    The pre-mix snapshot key is `scene_offset` (not Pointcept's `unmix3d_offset`)
    because the in-repo model code branches on `unmix3d_offset` in a way that
    presumes a different cam_offset shape; the losses already read `scene_offset`.
    """
    assert isinstance(batch[0], Mapping)

    # ---- Pre-collate: pad image_coord/image_mask to max_cam, prepare cam_offset
    if "image" in batch[0]:
        cam_counts = [int(s["image"].shape[0]) for s in batch]
        max_cam = max(cam_counts) if cam_counts else 0
        # A zero-cam sample may carry a `(0, C, 1, 1)` placeholder whose H/W don't
        # match real samples — reshape it to the canonical H/W before concat.
        ref = next((s["image"] for s in batch if s["image"].shape[0] > 0), None)
        for s, cam_i in zip(batch, cam_counts):
            if cam_i == 0 and ref is not None and s["image"].shape[1:] != ref.shape[1:]:
                s["image"] = s["image"].new_zeros(0, *ref.shape[1:])
            if cam_i < max_cam:
                pad = max_cam - cam_i
                ic = s["image_coord"]
                im = s["image_mask"]
                s["image_coord"] = torch.cat(
                    [ic, ic.new_zeros(ic.shape[0], pad, ic.shape[-1])], dim=1
                )
                s["image_mask"] = torch.cat(
                    [im, im.new_zeros(im.shape[0], pad)], dim=1
                )
            # `cam_offset` is given in diff form per sample; generic collate's
            # cumsum-on-"offset"-keys turns it into the cumulative form.
            s["cam_offset"] = torch.tensor([cam_i], dtype=torch.long)

    batch = collate_fn(batch)

    if random.random() >= mix_prob or "offset" not in batch or len(batch["offset"]) < 2:
        return batch

    offset = batch["offset"]
    batch["scene_offset"] = offset.clone()

    # ---- Image-aware compaction inside Mix3D (replaces old camera widening) ----
    if "image_coord" in batch and "cam_offset" in batch:
        cam_offset_cum = batch["cam_offset"]
        scene_starts_pts  = torch.cat([offset.new_zeros(1), offset[:-1]])
        scene_starts_cams = torch.cat([cam_offset_cum.new_zeros(1), cam_offset_cum[:-1]])
        cam_count = cam_offset_cum - scene_starts_cams   # (B,) per-scene CAM_i

        B       = len(offset)
        n_pairs = B // 2
        mega_cam_counts = [
            int(cam_count[2 * p] + cam_count[2 * p + 1]) for p in range(n_pairs)
        ]
        if B % 2 == 1:
            mega_cam_counts.append(int(cam_count[-1]))
        new_max_cam = max(mega_cam_counts) if mega_cam_counts else 0

        new_coord = batch["image_coord"].new_zeros(
            batch["image_coord"].shape[0], new_max_cam, batch["image_coord"].shape[-1]
        )
        new_mask = batch["image_mask"].new_zeros(
            batch["image_mask"].shape[0], new_max_cam
        )

        for p in range(n_pairs):
            sA, sB = 2 * p, 2 * p + 1
            psA, peA = int(scene_starts_pts[sA]), int(offset[sA])
            psB, peB = int(scene_starts_pts[sB]), int(offset[sB])
            cA, cB   = int(cam_count[sA]), int(cam_count[sB])
            new_coord[psA:peA, :cA]            = batch["image_coord"][psA:peA, :cA]
            new_coord[psB:peB, cA : cA + cB]   = batch["image_coord"][psB:peB, :cB]
            new_mask [psA:peA, :cA]            = batch["image_mask"] [psA:peA, :cA]
            new_mask [psB:peB, cA : cA + cB]   = batch["image_mask"] [psB:peB, :cB]
        if B % 2 == 1:
            last = B - 1
            ps, pe = int(scene_starts_pts[last]), int(offset[last])
            cl     = int(cam_count[last])
            new_coord[ps:pe, :cl] = batch["image_coord"][ps:pe, :cl]
            new_mask [ps:pe, :cl] = batch["image_mask"] [ps:pe, :cl]

        batch["image_coord"] = new_coord
        batch["image_mask"]  = new_mask
        # `image` is already in scene order; pair-merging cam_offset (below)
        # makes per-mega-scene slices into the packed image tensor correct.
        
    for key in [k for k in batch if "offset" in k and k != "scene_offset"]:
        batch[key] = torch.cat(
            [batch[key][1:-1:2], batch[key][-1].unsqueeze(0)], dim=0
        )

    return batch


def points2image(
    points: np.ndarray,  # (N, 3)
    points2cam: np.ndarray,  # (4, 4)
    K: np.ndarray,  # (3, 3)
    image_size: tuple[int, int],  # (H, W)
    depth: np.ndarray = None,  # (HxW,)
    min_distance: float = None,
    max_distance: float = None,
    error_margin: float = None,
    return_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:  # (N, 2), (N,)  [+ (N,) per-point camera depth]
    # global to camera coords
    points_cam = points @ points2cam[:3, :3].T + points2cam[:3, 3]

    # points in front of camera
    visibility_mask = points_cam[:, 2] > 0

    # project
    points_image = points_cam @ K.T
    points_image[:, :2] = points_image[:, :2] / points_image[:, 2:]

    # within image size
    visibility_mask = np.logical_and(visibility_mask, np.all(points_image >= 0, axis=1))
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 0] < image_size[1]
    )
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 1] < image_size[0]
    )
    if min_distance is not None:
        visibility_mask = np.logical_and(
            visibility_mask, points_image[:, 2] > min_distance
        )
    if max_distance is not None:
        visibility_mask = np.logical_and(
            visibility_mask, points_image[:, 2] < max_distance
        )
    if depth is not None and error_margin is not None:
        points_depth = depth[
            points_image[visibility_mask][:, 1].astype(int),
            points_image[visibility_mask][:, 0].astype(int),
        ]

        visibility_mask[visibility_mask] = (
            np.abs(points_image[visibility_mask][:, 2] - points_depth) <= error_margin
        )

    if return_depth:
        # points_image[:, 2] is the camera-frame depth (z along the optical axis),
        # before the perspective divide above overwrote only columns 0:2.
        return points_image[:, :2], visibility_mask, points_image[:, 2]
    return points_image[:, :2], visibility_mask