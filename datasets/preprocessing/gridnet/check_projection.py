"""
check_projection.py

Visual sanity-check of the projection pipeline on real data.

If the projection is correct, the dots in overlay.jpg should sit on the
objects they belong to, and the colour error should be small.

Usage:

python check_projection.py --zone-dir  PATH/TO/t1z4
                            --out-dir   /tmp/proj_check
                            [--cam-idx  0]
                            [--n-pts    20000]
"""

import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image


# ── inlined from gridnet_utils (avoids the JAX import at module level) ────────

def parse_calibration_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    cal  = dict(width=0, height=0, f=0.0, cx=0.0, cy=0.0,
                k1=0.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0,
                p1=0.0, p2=0.0, p3=0.0, p4=0.0, b1=0.0, b2=0.0)
    for el in root:
        if el.tag not in cal:
            continue
        cal[el.tag] = int(el.text) if el.tag in ('width', 'height') else float(el.text)
    return cal


def read_camera_file(filepath):
    """Reads camera poses. Positions are in the same absolute world space as las.x."""
    data = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            v = line.strip().split()
            if len(v) < 7:
                continue
            data[v[0]] = dict(
                Xs    = float(v[1]),
                Ys    = float(v[2]),
                Zs    = float(v[3]),
                omega = np.radians(float(v[4])),
                phi   = np.radians(float(v[5])),
                kappa = np.radians(float(v[6])),
            )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy projection (mirrors gridnet_utils without the JAX dependency)
# ─────────────────────────────────────────────────────────────────────────────

def rot_zyx(o, p, k):
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(o), np.sin(o)],
                   [0, -np.sin(o), np.cos(o)]])
    Ry = np.array([[ np.cos(p), 0, -np.sin(p)],
                   [0,          1,  0],
                   [ np.sin(p), 0,  np.cos(p)]])
    Rz = np.array([[ np.cos(k), np.sin(k), 0],
                   [-np.sin(k), np.cos(k), 0],
                   [0,          0,          1]])
    return Rz @ Ry @ Rx


def project_points(xyz, cam, cal):
    """
    Vectorised Agisoft projection.

    xyz : (N, 3) float64  — absolute world coords (las.X * scale + header.offset)
    cam : dict  — Xs, Ys, Zs in absolute world coords, omega/phi/kappa in radians
    cal : dict  — width, height, f, cx, cy, k1..k5, p1..p4, b1, b2

    Returns
    -------
    fx, fy   : (N,) float64 — pixel coords in the original (full-resolution) image
    z        : (N,) float64 — depth (positive = in front of camera)
    visible  : (N,) bool    — in-frustum AND z > 0 AND within image bounds
    """
    R = rot_zyx(cam['omega'], cam['phi'], cam['kappa'])
    S = np.array([cam['Xs'], cam['Ys'], cam['Zs']])

    # camera-frame coordinates
    cam_xyz = (xyz - S) @ R.T          # (N, 3)

    z  = -cam_xyz[:, 2]               # depth (positive = in front)
    ok = z > 0

    # normalised image coords (pinhole)
    x = np.where(ok, -cam_xyz[:, 0] / cam_xyz[:, 2], 0.0)
    y = np.where(ok, -cam_xyz[:, 1] / cam_xyz[:, 2], 0.0)

    # in-frustum check (before Agisoft y-flip)
    w2 = cal['width']  / cal['f'] / 2
    h2 = cal['height'] / cal['f'] / 2
    in_frustum = ok & (x >= -w2) & (x < w2) & (y >= -h2) & (y < h2)

    # Agisoft y-flip
    y = -y

    # radial + tangential distortion
    rc  = x**2 + y**2
    dr  = (1
           + cal['k1']*rc
           + cal['k2']*rc**2
           + cal['k3']*rc**3
           + cal['k4']*rc**4
           + cal['k5']*rc**5)
    xp = x*dr + cal['p1']*(rc + 2*x**2) + 2*cal['p2']*x*y*(1 + cal['p3']*rc + cal['p4']*rc**2)
    yp = y*dr + cal['p2']*(rc + 2*y**2) + 2*cal['p1']*x*y*(1 + cal['p3']*rc + cal['p4']*rc**2)

    fx = cal['width']  * 0.5 + cal['cx'] + xp*cal['f'] + xp*cal['b1'] + yp*cal['b2']
    fy = cal['height'] * 0.5 + cal['cy'] + yp*cal['f']

    # final image-bounds check
    in_image = (in_frustum &
                (fx >= 0) & (fx < cal['width']) &
                (fy >= 0) & (fy < cal['height']))

    return fx, fy, z, in_image


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_las_rgb(las_path, n_pts):
    """Loads xyz in absolute world coordinates (float64) and rgb (uint8)."""
    import laspy
    las = laspy.read(las_path)

    # Absolute world coordinates: X * scale + offset  (float64, no precision loss)
    xyz = np.vstack((las.X, las.Y, las.Z)).T * las.header.scale + las.header.offset

    # 16-bit RGB → uint8
    rgb = np.stack([
        np.array(las.red,   dtype=np.float32),
        np.array(las.green, dtype=np.float32),
        np.array(las.blue,  dtype=np.float32),
    ], axis=1)
    rgb = (rgb / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)

    # random subset
    if n_pts < len(xyz):
        idx = np.random.choice(len(xyz), n_pts, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


# ─────────────────────────────────────────────────────────────────────────────
# Output images
# ─────────────────────────────────────────────────────────────────────────────

def _circle_offsets(r):
    """Returns (dy, dx) arrays of pixel offsets that form a filled circle of radius r."""
    g = np.mgrid[-r:r+1, -r:r+1]
    mask = g[0]**2 + g[1]**2 <= r**2
    dy, dx = np.where(mask)
    return dy - r, dx - r


def save_painted(image, fx, fy, visible, las_rgb_vis, out_path, dot_radius=4):
    """
    Original image with LAS-coloured dots at each projected position.
    Each dot is painted with the actual LAS RGB colour of that point.
    """
    arr = np.array(image, dtype=np.uint8)
    H, W = arr.shape[:2]

    u = fx[visible].astype(np.int32)   # (N_vis,)
    v = fy[visible].astype(np.int32)

    dy, dx = _circle_offsets(dot_radius)  # (K,)

    # For each point: expand to all circle pixels
    rows = (v[:, None] + dy[None, :]).ravel()   # (N_vis * K,)
    cols = (u[:, None] + dx[None, :]).ravel()
    # Repeat each point's colour K times
    colors = np.repeat(las_rgb_vis, len(dy), axis=0)  # (N_vis * K, 3)

    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    arr[rows[valid], cols[valid]] = colors[valid]

    # downscale to ~3000px wide for manageable file size
    scale = min(1.0, 3000 / W)
    out_w, out_h = int(W * scale), int(H * scale)
    Image.fromarray(arr).resize((out_w, out_h), Image.BILINEAR).save(out_path, quality=92)
    print(f"  painted       → {out_path}  ({visible.sum():,} pts, r={dot_radius}px, LAS colours)")


def save_reprojected(image, fx, fy, visible, las_rgb_vis, out_path, dot_radius=4):
    """
    Side-by-side: left=original photo, right=black canvas with LAS colours at projected positions.
    Uses the same dot radius as save_painted so dots are visible despite sparse LiDAR coverage.
    Note: LiDAR is much sparser than the image (~0.2% pixel coverage), so the right panel
    will always look sparse — but colours and positions should match the original.
    """
    arr_orig = np.array(image, dtype=np.uint8)
    H, W = arr_orig.shape[:2]

    reproj = np.zeros_like(arr_orig)
    u = fx[visible].astype(np.int32)
    v = fy[visible].astype(np.int32)

    dy, dx = _circle_offsets(dot_radius)
    rows   = (v[:, None] + dy[None, :]).ravel()
    cols   = (u[:, None] + dx[None, :]).ravel()
    colors = np.repeat(las_rgb_vis, len(dy), axis=0)
    valid  = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    reproj[rows[valid], cols[valid]] = colors[valid]

    # side-by-side: left=original, right=reprojected
    panel = np.concatenate([arr_orig, reproj], axis=1)
    scale = min(1.0, 3000 / W)
    out_w, out_h = int(W * scale), int(H * scale)
    Image.fromarray(panel).resize((out_w * 2, out_h), Image.BILINEAR).save(out_path, quality=92)
    print(f"  reprojected   → {out_path}  (left=original, right=LAS colours — sparse by nature)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(zone_dir, out_dir, cam_idx, n_pts, seed):
    np.random.seed(seed)
    zone_dir = Path(zone_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── calibration ──────────────────────────────────────────────────────────
    cal = parse_calibration_xml(zone_dir / 'pose' / 'camera_calibration.xml')
    print(f"Calibration: {cal['width']}×{cal['height']}  f={cal['f']:.1f}")

    # ── LAS ──────────────────────────────────────────────────────────────────
    las_path = next((zone_dir / 'lidar').glob('*.las'))
    print(f"Loading {n_pts:,} points from {las_path.name} ...")
    xyz, las_rgb = load_las_rgb(las_path, n_pts)

    # ── camera poses ─────────────────────────────────────────────────────────
    # Both xyz and camera positions are in the same absolute world space (las.x convention).
    cameras = read_camera_file(zone_dir / 'pose' / 'cameras_pose.txt')
    cam_names = list(cameras.keys())
    print(f"{len(cam_names)} cameras found")

    cam_name = cam_names[cam_idx]
    cam      = cameras[cam_name]
    print(f"Using camera [{cam_idx}]: {cam_name}")
    print(f"  position  Xs={cam['Xs']:.2f}  Ys={cam['Ys']:.2f}  Zs={cam['Zs']:.2f}")
    print(f"  angles    ω={np.degrees(cam['omega']):.2f}°  "
          f"φ={np.degrees(cam['phi']):.2f}°  "
          f"κ={np.degrees(cam['kappa']):.2f}°")

    # ── project ───────────────────────────────────────────────────────────────
    print("Projecting ...")
    fx, fy, z, visible = project_points(xyz, cam, cal)
    print(f"  {visible.sum():,} / {len(xyz):,} points visible in this camera")

    if visible.sum() == 0:
        print("No points visible — try a different camera index (--cam-idx).")
        return

    # ── load image ────────────────────────────────────────────────────────────
    # Try common extensions
    for ext in ['.JPG', '.jpg', '.jpeg', '.PNG', '.png']:
        img_path = zone_dir / 'images' / (cam_name + ext)
        if img_path.exists():
            break
    else:
        img_path = zone_dir / 'images' / cam_name     # name already has extension
        if not img_path.exists():
            print(f"Image not found for camera '{cam_name}'. "
                  f"Check that the image folder contains the right files.")
            return

    print(f"Loading image {img_path.name} ...")
    image = Image.open(img_path).convert('RGB')
    img_arr = np.array(image)   # (H, W, 3) uint8

    # ── sample image colours at projected positions ───────────────────────────
    u = fx[visible].astype(np.int32).clip(0, cal['width']  - 1)
    v = fy[visible].astype(np.int32).clip(0, cal['height'] - 1)
    img_rgb_at_pts = img_arr[v, u]       # (N_vis, 3) uint8

    las_rgb_vis = las_rgb[visible]        # (N_vis, 3) uint8

    # ── colour error ──────────────────────────────────────────────────────────
    diff = las_rgb_vis.astype(np.float32) - img_rgb_at_pts.astype(np.float32)
    mae  = np.abs(diff).mean(axis=0)
    print(f"\nMean absolute colour error (0–255 scale):")
    print(f"  R: {mae[0]:.1f}   G: {mae[1]:.1f}   B: {mae[2]:.1f}   "
          f"mean: {mae.mean():.1f}")
    if mae.mean() < 25:
        print("  ✓ Error looks good — projection is likely correct.")
    elif mae.mean() < 50:
        print("  ⚠ Moderate error — check for colour correction differences "
              "between LAS and image.")
    else:
        print("  ✗ Large error — projection may be wrong. "
              "Check offset, angle convention or filename matching.")

    # ── save outputs ──────────────────────────────────────────────────────────
    save_painted(image, fx, fy, visible, las_rgb_vis, out_dir / 'painted.jpg')
    save_reprojected(image, fx, fy, visible, las_rgb_vis, out_dir / 'reprojected.jpg')
    print(f"\nDone. Check {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual projection check on real data')
    parser.add_argument('--zone-dir', required=True,
                        help='Path to a zone folder (e.g. /data/GridNet-HD/t1z4)')
    parser.add_argument('--out-dir',  required=True,
                        help='Directory for output images')
    parser.add_argument('--cam-idx',  type=int, default=0,
                        help='Index of the camera to use (default: 0)')
    parser.add_argument('--n-pts',    type=int, default=100_000,
                        help='Number of LAS points to sample (default: 100 000)')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    run(args.zone_dir, args.out_dir, args.cam_idx, args.n_pts, args.seed)
