# WSL: export DISPLAY=:0 && export XDG_SESSION_TYPE=x11
import argparse
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

# ── GridNet 12-class colormap (train IDs 0-11) ────────────────────────────────
CLASS_NAMES = [
    "pylon",        # 0
    "conductor",    # 1
    "structural",   # 2
    "insulator",    # 3
    "high_veg",     # 4
    "low_veg",      # 5
    "herb",         # 6
    "gravel",       # 7
    "impervious",   # 8
    "water",        # 9
    "building",     # 10
    "unlabeled",    # 11 / ignore
]

# RGB in [0, 255]
CLASS_COLORS = np.array([
    [210, 210, 210],   #  0 pylon        light gray
    [255, 255,   0],   #  1 conductor    yellow
    [160,  82,  45],   #  2 structural   brown
    [  0, 255, 255],   #  3 insulator    cyan
    [  0, 100,   0],   #  4 high_veg     dark green
    [144, 238, 144],   #  5 low_veg      light green
    [ 50, 205,  50],   #  6 herb         lime
    [160, 160, 160],   #  7 gravel       mid-gray
    [ 80,  80,  80],   #  8 impervious   dark gray
    [  0,   0, 255],   #  9 water        blue
    [220,  50,  50],   # 10 building     red
    [255, 140,   0],   # 11 unlabeled    orange
], dtype=np.float32) / 255.0


# helpers

def _load_las(las_path):
    """Return xyz (N,3) float64, rgb (N,3) float32 [0-255], labels or None."""
    import laspy
    las = laspy.read(las_path)
    xyz = np.vstack((las.X, las.Y, las.Z)).T * las.header.scale + las.header.offset
    rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32) / 65535 * 255
    labels = None
    if "ground_truth" in las.point_format.extra_dimension_names:
        labels = np.array(las["ground_truth"], dtype=np.int32)
    elif "classification" in las.point_format.dimension_names:
        labels = np.array(las["classification"], dtype=np.int32)
    return xyz, rgb, labels


def _voxel_downsample(pts, colors, labels, voxel_size):
    """Keep one representative point per voxel (first-point rule)."""
    vox = np.floor(pts / voxel_size).astype(np.int64)
    vox -= vox.min(axis=0)
    mx = vox.max(axis=0) + 1
    keys = vox[:, 0] * (mx[1] * mx[2]) + vox[:, 1] * mx[2] + vox[:, 2]
    _, keep = np.unique(keys, return_index=True)
    keep.sort()
    return pts[keep], colors[keep], (labels[keep] if labels is not None else None)


def _labels_to_colors(labels):
    """Map integer label array → (N,3) float32 RGB in [0,1]."""
    colors = np.full((len(labels), 3), CLASS_COLORS[-1], dtype=np.float32)
    for cls_id in range(len(CLASS_NAMES)):
        mask = labels == cls_id
        if mask.any():
            colors[mask] = CLASS_COLORS[cls_id]
    return colors


def _print_legend(labels):
    """Print class distribution to console."""
    print("\nClass distribution:")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = int((labels == cls_id).sum())
        if count > 0:
            pct = count / len(labels) * 100
            print(f"  [{cls_id:2d}] {name:<14s} {count:>10,}  ({pct:.2f}%)")
    print()


def _make_pcd(xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _show(xyz, colors, title=""):
    """Display point cloud in Open3D interactive viewer."""
    pcd = _make_pcd(xyz, colors)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    o3d.visualization.draw_geometries([pcd, frame], window_name=title)


def _render_offscreen(xyz, colors, out_path, width=1920, height=1080):
    """Render point cloud to a PNG using Open3D offscreen renderer (no display needed)."""
    pcd = _make_pcd(xyz, colors)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.15, 0.15, 0.15, 1.0])  # dark background

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0
    renderer.scene.add_geometry("pcd", pcd, mat)

    # Fit camera to the bounding box
    bounds = pcd.get_axis_aligned_bounding_box()
    renderer.setup_camera(60.0, bounds, bounds.get_center())

    img = renderer.render_to_image()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out_path), img)
    print(f"Saved → {out_path}")


# public API

def vizualize_point_cloud_with_colors(points, rgb):
    """Legacy helper — show a point cloud given explicit colours."""
    _show(points, rgb)


def visualize_predictions(las_path, npz_path, voxel_size=0.1, show_rgb=False,
                          offscreen=False, out_path=None, width=1920, height=1080):
    """
    Overlay NPZ predictions on the original LAS point cloud.

    Parameters
    ----------
    las_path   : path to the original LAS file (for coordinates)
    npz_path   : path to the .npz prediction file (key 'data', uint8 labels)
    voxel_size : downsampling resolution for display (metres); 0 = no downsampling
    show_rgb   : if True, show original RGB colours alongside predictions
    """
    print(f"Loading LAS  : {las_path}")
    xyz, rgb, _ = _load_las(las_path)
    print(f"  {len(xyz):,} points")

    print(f"Loading NPZ  : {npz_path}")
    pred = np.load(npz_path)["data"].astype(np.int32)
    print(f"  {len(pred):,} labels")

    if len(xyz) != len(pred):
        raise ValueError(
            f"Point count mismatch: LAS has {len(xyz):,} points "
            f"but NPZ has {len(pred):,} labels."
        )

    _print_legend(pred)

    pred_colors = _labels_to_colors(pred)

    if voxel_size > 0:
        print(f"Downsampling at {voxel_size} m …")
        xyz_d, pred_colors_d, _ = _voxel_downsample(xyz, pred_colors, pred, voxel_size)
        print(f"  {len(xyz_d):,} points after downsampling\n")
    else:
        xyz_d, pred_colors_d = xyz, pred_colors

    # Centre for nicer navigation
    xyz_d = xyz_d - xyz_d.mean(axis=0)

    area = Path(npz_path).stem

    if offscreen:
        pred_out = Path(out_path) if out_path else Path(npz_path).with_suffix(".png")
        _render_offscreen(xyz_d, pred_colors_d, pred_out, width=width, height=height)
        if show_rgb:
            rgb_norm = rgb / 255.0
            if voxel_size > 0:
                _, rgb_d, _ = _voxel_downsample(xyz, rgb_norm, None, voxel_size)
            else:
                rgb_d = rgb_norm
            rgb_out = pred_out.with_name(pred_out.stem + "_rgb.png")
            _render_offscreen(xyz_d, rgb_d, rgb_out, width=width, height=height)
    else:
        print("Opening viewer — press Q to quit.")
        _show(xyz_d, pred_colors_d, title=f"Predictions — {area}")
        if show_rgb:
            rgb_norm = rgb / 255.0
            if voxel_size > 0:
                _, rgb_d, _ = _voxel_downsample(xyz, rgb_norm, None, voxel_size)
            else:
                rgb_d = rgb_norm
            print("Opening RGB viewer — press Q to quit.")
            _show(xyz_d, rgb_d, title=f"RGB — {area}")


# entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise NPZ predictions overlaid on the original LAS file."
    )
    parser.add_argument("--las",    required=True, help="Original LAS file")
    parser.add_argument("--npz",    required=True, help="NPZ prediction file (key 'data')")
    parser.add_argument("--voxel",  type=float, default=0.1,
                        help="Display voxel size in metres (default 0.1; 0 = no downsampling)")
    parser.add_argument("--rgb",      action="store_true",
                        help="Also render original RGB colours for comparison")
    parser.add_argument("--offscreen", action="store_true",
                        help="Offscreen rendering to PNG — no display required (SLURM-safe)")
    parser.add_argument("--out",      default=None,
                        help="Output PNG path for offscreen mode (default: <npz_stem>.png)")
    parser.add_argument("--width",    type=int, default=1920, help="Output image width")
    parser.add_argument("--height",   type=int, default=1080, help="Output image height")
    args = parser.parse_args()

    if o3d is None:
        raise ImportError("open3d is required: pip install open3d")

    visualize_predictions(
        las_path=args.las,
        npz_path=args.npz,
        voxel_size=args.voxel,
        show_rgb=args.rgb,
        offscreen=args.offscreen,
        out_path=args.out,
        width=args.width,
        height=args.height,
    )
