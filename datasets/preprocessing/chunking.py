import json
from pathlib import Path
import numpy as np
import torch
import laspy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

ID2TRAINID = np.asarray([0,0,0,0,0,1,2,2,3,3,3,
			 3,11,11,4,5,6,7,7,8,9,
			 10,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11,11,11,11,11,11,
			 11,11,11,11,11])


def pair(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

def voxel_downsample(coord, feat=None, label=None, voxel_size=0.05):
    """
    Simple voxel grid downsampling
    """

    voxel = np.floor(coord / voxel_size).astype(np.int64)

    _, unique_idx = np.unique(voxel, axis=0, return_index=True)

    coord = coord[unique_idx]

    if feat is not None:
        feat = feat[unique_idx]

    if label is not None:
        label = label[unique_idx]

    return coord, feat, label

def chunk_scene_fast(
    las_path,
    out_dir,
    block_size=(50, 50),
    stride=(40, 40),
    min_points=1024,
    voxel_size=None,
    rgb=True,
    intensity=True,
    use_ground_truth=True,
    remap=True,
):

    las = laspy.read(las_path)

    coords = np.vstack((las.X, las.Y, las.Z)).T
    coords = coords * las.header.scale# + las.header.offset

    features_list = []

    if rgb:
        rgb_feat = np.stack(
            [las.red, las.green, las.blue],
            axis=1
        ).astype(np.float32) / 65535
        rgb_feat *= 255
        features_list.append(rgb_feat)

    if intensity:
        inten_feat = las.intensity.astype(np.float32).clip(0, 60000) / 60000
        features_list.append(inten_feat[:, None])

    feat = np.concatenate(features_list, axis=1) if features_list else None

    if use_ground_truth and "ground_truth" in las.point_format.extra_dimension_names:
        label = las["ground_truth"].astype(np.int64)
        if remap:
            label = ID2TRAINID[label]
    else:
        label = None

    min_coord = coords.min(axis=0)

    # Precompute grid indices for fast candidate search

    coords = coords - min_coord
    max_coord = coords.max(axis=0)

    grid_x = np.floor(coords[:, 0] / stride[0]).astype(np.int32)
    grid_y = np.floor(coords[:, 1] / stride[1]).astype(np.int32)

    grid_id = grid_x * 100000 + grid_y

    order = np.argsort(grid_id)

    coords = coords[order]
    grid_x = grid_x[order]
    grid_y = grid_y[order]

    if feat is not None:
        feat = feat[order]

    if label is not None:
        label = label[order]


    x_starts = np.arange(0, max_coord[0], stride[0])
    y_starts = np.arange(0, max_coord[1], stride[1])

    counter = 0
    zone = las_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for x in x_starts:
        for y in y_starts:

            gx_min = int(x // stride[0])
            gy_min = int(y // stride[1])

            gx_max = int((x + block_size[0]) // stride[0])
            gy_max = int((y + block_size[1]) // stride[1])

            # fast candidate filtering using grid cells
            cell_mask = (
                (grid_x >= gx_min) & (grid_x <= gx_max) &
                (grid_y >= gy_min) & (grid_y <= gy_max)
            )

            candidate_idx = np.where(cell_mask)[0]

            if candidate_idx.size == 0:
                continue

            pts_candidate = coords[candidate_idx]

            block_mask = (
                (pts_candidate[:, 0] >= x) & (pts_candidate[:, 0] <= x + block_size[0]) &
                (pts_candidate[:, 1] >= y) & (pts_candidate[:, 1] <= y + block_size[1])
            )

            idx = candidate_idx[block_mask]

            if len(idx) < min_points:
                continue

            pts = coords[idx]
            f = feat[idx] if feat is not None else None
            l = label[idx] if label is not None else None

            if voxel_size:
                pts, f, l = voxel_downsample(pts, f, l, voxel_size)

            colors = None
            inten = None

            if f is not None:
                if rgb:
                    colors = f[:, :3].astype(np.float32)
                if intensity:
                    inten = f[:, -1].astype(np.float32)

            data = {
                "coord": pts.astype(np.float32),
                "min_coord_offset": min_coord.astype(np.float32),
                "color": colors if colors is not None else None,
                "intensity": inten if inten is not None else None,
                "segment": l.astype(np.int32) if l is not None else None,
            }

            save_path = out_dir / f"{zone}_{counter:06d}.npz"
            if not save_path.exists():
                np.savez_compressed(save_path, **data)  # saves all arrays inside the dictionary
            counter += 1

def process_zone(args):
    zone, root_dir, split_out, block_size, stride, min_points, voxel_size = args

    las_path = root_dir / zone / "lidar" / f"{zone}.las"

    chunk_scene_fast(
        las_path,
        split_out,
        block_size,
        stride,
        min_points,
        voxel_size
    )

def main(
    root_dir,
    split_json,
    output_dir,
    split = "train",
    block_size=(50, 50),
    stride=(50, 50),
    min_points=1024,
    voxel_size=None,
    num_workers=None
):

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    block_size = pair(block_size)
    stride = pair(stride)

    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    with open(split_json) as f:
        splits = json.load(f)

    for split_key, zones in splits.items():

        if not split_key == split:
            continue
        

        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {split} with {num_workers} workers")

        args_list = [
            (
                zone,
                root_dir,
                split_out,
                block_size,
                stride,
                min_points,
                voxel_size
            )
            for zone in zones
        ]

        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(process_zone, args_list),
                total=len(args_list)
            ))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk LAS scenes into fixed-size .npz blocks.")
    parser.add_argument("--root-dir",    required=True,       help="Root directory containing zone folders")
    parser.add_argument("--split-json",  required=True,       help="Path to split.json")
    parser.add_argument("--output-dir",  required=True,       help="Output directory for chunks")
    parser.add_argument("--split",       default="train",     help="Split to process: train | val | test (default: train)")
    parser.add_argument("--block-size",  type=float, default=15.0,  help="Block size in meters (default: 15)")
    parser.add_argument("--stride",      type=float, default=8.0,   help="Stride in meters (default: 8)")
    parser.add_argument("--min-points",  type=int,   default=10000, help="Minimum points per chunk (default: 10000)")
    parser.add_argument("--voxel-size",  type=float, default=None,  help="Optional voxel downsampling size in meters")
    parser.add_argument("--num-workers", type=int,   default=None,  help="Number of parallel workers (default: cpu_count-1)")
    args = parser.parse_args()

    main(
        root_dir=args.root_dir,
        split_json=args.split_json,
        output_dir=args.output_dir,
        split=args.split,
        block_size=args.block_size,
        stride=args.stride,
        min_points=args.min_points,
        voxel_size=args.voxel_size,
        num_workers=args.num_workers,
    )