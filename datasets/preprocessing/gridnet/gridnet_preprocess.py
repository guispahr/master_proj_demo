"""
gridnet_preprocess.py

Preprocessing pipeline for GridNet-HD.

Output layout:
    out_root/{split}/{zone}/
    ├── images/ 
    └── chunk_XXXXX/
        ├── xyz.npy          (N, 3)     float32  local coords (relative to chunk centroid)
        ├── rgb.npy          (N, 3)     float32  LAS RGB in [0, 255]
        ├── labels.npy       (N,)       uint8    remapped train IDs (absent for test set)
        ├── image_coord.npz  (N, C, 2)  int16    (u,v) in resized image; -1 = invisible (compressed)
        └── meta.json        {origin, tile_center, bbox, cam_names}

Usage:
    python gridnet_preprocess.py \\
        --raw-root   /data/GridNet-HD \\
        --out-root   /data/preprocessed \\
        --split-json /data/GridNet-HD/split.json \\
        --split      train \\
        --target-h   518 --target-w 518 \\
        --workers    4
"""

import argparse
import json
import laspy
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from scipy.spatial import cKDTree
from scipy.ndimage import minimum_filter
import xml.etree.ElementTree as ET
from PIL import Image
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# Label remapping: raw LAS class ID -> training ID (mirrors chunking.py)
ID2TRAINID = np.asarray([
     0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3,
     3,11,11, 4, 5, 6, 7, 7, 8, 9,
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
    11,11,11,11,11,
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_calibration(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cal  = dict(width=0, height=0, f=0.0, cx=0.0, cy=0.0,
                k1=0.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0,
                p1=0.0, p2=0.0, p3=0.0, p4=0.0, b1=0.0, b2=0.0)
    for el in root:
        if el.tag in cal:
            cal[el.tag] = int(el.text) if el.tag in ('width', 'height') else float(el.text)
    return cal


def _read_cameras(filepath):
    """Returns {name: {Xs,Ys,Zs (absolute world coords), omega,phi,kappa (radians)}}."""
    data = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            v = line.strip().split()
            if len(v) < 7:
                continue
            data[v[0]] = dict(
                Xs=float(v[1]), Ys=float(v[2]), Zs=float(v[3]),
                omega=np.radians(float(v[4])),
                phi=np.radians(float(v[5])),
                kappa=np.radians(float(v[6])),
            )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_zone_points(las_path):
    """
    Returns:
        xyz       : (N, 3) float64  absolute world coords (X*scale + offset)
        rgb       : (N, 3) float32  LAS RGB in [0, 255]
        intensity : (N,)   float32  normalised to [0, 1]
        labels    : (N,)   int32    raw class IDs, or None if absent (test set)
    """
    las = laspy.read(las_path)
    xyz       = np.vstack((las.X, las.Y, las.Z)).T * las.header.scale + las.header.offset
    rgb       = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32) / 65535 * 255
    intensity = las.intensity.astype(np.float32).clip(0, 255.0) / 255.0
    if 'ground_truth' in las.point_format.extra_dimension_names:
        labels = np.array(las['ground_truth'], dtype=np.int32)
    else:
        labels = None
    return xyz, rgb, intensity, labels


# ─────────────────────────────────────────────────────────────────────────────
# Chunking — grid-indexed, no full-N mask per tile
# ─────────────────────────────────────────────────────────────────────────────

def _iter_chunks(xyz, rgb, intensity, labels, chunk_size, stride, min_pts):
    """
    Generator yielding one chunk dict at a time (sliding-window, fixed size).

    Windows of exactly chunk_size × chunk_size metres are placed on a grid
    with spacing `stride`.  stride < chunk_size gives overlap between neighbours.

    Pre-sorts points into a spatial grid so each tile only scans its
    neighbouring cells — avoids creating 50M-element boolean masks.

    Yields:
        pts           : (N_k, 3) float64  absolute coords
        rgb           : (N_k, 3) float32
        intensity     : (N_k,)   float32
        labels        : (N_k,)   int32 or None
        tile_center   : (2,)     float64  window centre [x+cs/2, y+cs/2]
        origin        : (3,)     float64  point cloud centroid (local coord origin)
        bbox          : (x0, y0, x1, y1)  window bounds
        idx_in_input  : (N_k,)   int64    index of each chunk point into the *input*
                        xyz array (before the internal spatial sort).  Used by
                        preprocess_zone to construct orig_idx.npy.
    """
    xmin = xyz[:, 0].min()
    ymin = xyz[:, 1].min()

    gx = np.floor((xyz[:, 0] - xmin) / chunk_size).astype(np.int32)
    gy = np.floor((xyz[:, 1] - ymin) / chunk_size).astype(np.int32)

    order = np.argsort(gx * 1_000_000 + gy, kind='stable')
    xyz_s = xyz[order]
    rgb_s = rgb[order]
    int_s = intensity[order]
    lab_s = labels[order] if labels is not None else None
    gx_s  = gx[order]
    gy_s  = gy[order]
    del gx, gy

    cell_id      = gx_s * 1_000_000 + gy_s
    uniq, starts = np.unique(cell_id, return_index=True)
    ends         = np.append(starts[1:], len(cell_id))
    cell_map     = {int(u): (int(s), int(e))
                    for u, s, e in zip(uniq, starts, ends)}
    del cell_id, uniq, starts, ends

    xmax = xyz_s[:, 0].max()
    ymax = xyz_s[:, 1].max()

    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            cx0 = max(0, int(np.floor((x            - xmin) / chunk_size)))
            cx1 =        int(np.floor((x + chunk_size - xmin) / chunk_size))
            cy0 = max(0, int(np.floor((y            - ymin) / chunk_size)))
            cy1 =        int(np.floor((y + chunk_size - ymin) / chunk_size))

            parts = [cell_map[k]
                     for cx in range(cx0, cx1 + 1)
                     for cy in range(cy0, cy1 + 1)
                     if (k := cx * 1_000_000 + cy) in cell_map]

            if parts:
                idx   = np.concatenate([np.arange(s, e) for s, e in parts])
                pts_c = xyz_s[idx]
                fine  = ((pts_c[:, 0] >= x) &
                         (pts_c[:, 0] <  x + chunk_size) &
                         (pts_c[:, 1] >= y) &
                         (pts_c[:, 1] <  y + chunk_size))
                idx   = idx[fine]
                if len(idx) >= min_pts:
                    pts = xyz_s[idx]
                    yield {
                        'pts':          pts,
                        'rgb':          rgb_s[idx],
                        'intensity':    int_s[idx],
                        'labels':       lab_s[idx] if lab_s is not None else None,
                        'tile_center':  np.array([x + chunk_size / 2,
                                                  y + chunk_size / 2]),
                        'origin':       pts.mean(axis=0),
                        'bbox':         (x, y, x + chunk_size, y + chunk_size),
                        'idx_in_input': order[idx],  # index into pre-sort xyz
                    }
            y += stride
        x += stride


# ─────────────────────────────────────────────────────────────────────────────
# Voxel downsampling
# ─────────────────────────────────────────────────────────────────────────────

def _voxel_downsample(pts, rgb, intensity, labels, voxel_size):
    """Keep one point per voxel (first-point rule, O(N log N)).
    pts, rgb, intensity, labels are in the same row order.
    Returns (downsampled arrays..., keep) where keep is the index array into the
    input pts that was retained, so pts[keep] == returned pts."""
    vox = np.floor(pts / voxel_size).astype(np.int64)
    vox -= vox.min(axis=0)
    mx = vox.max(axis=0) + 1
    keys = vox[:, 0] * (mx[1] * mx[2]) + vox[:, 1] * mx[2] + vox[:, 2]
    _, keep = np.unique(keys, return_index=True)
    keep.sort()
    return pts[keep], rgb[keep], intensity[keep], (labels[keep] if labels is not None else None), keep


# ─────────────────────────────────────────────────────────────────────────────
# Projection — JAX JIT, batched to avoid recompilation
# ─────────────────────────────────────────────────────────────────────────────

# Calibration dict keys → packed array order (must match _project_jax indexing)
_CAL_KEYS = ('width', 'height', 'f', 'cx', 'cy',
             'k1', 'k2', 'k3', 'k4', 'k5',
             'p1', 'p2', 'p3', 'p4', 'b1', 'b2')


def cal_to_jax(cal):
    """Pack calibration dict into a float64 JAX array (16,)."""
    return jnp.array([cal[k] for k in _CAL_KEYS], dtype=jnp.float64)


@jax.jit
def _project_jax(pts, cam, cal):
    """
    Agisoft projection — JIT-compiled, fixed batch size B for reuse.
    pts : (B, 3) float64 — absolute world coords
    cam : (6,)  float64 — [Xs, Ys, Zs, omega, phi, kappa]
    cal : (16,) float64 — packed calibration (see _CAL_KEYS)
    Returns fx, fy, z, ins — each (B,).
    """
    co, so = jnp.cos(cam[3]), jnp.sin(cam[3])
    cp, sp = jnp.cos(cam[4]), jnp.sin(cam[4])
    ck, sk = jnp.cos(cam[5]), jnp.sin(cam[5])

    Rz = jnp.array([[ ck, sk, 0.], [-sk, ck, 0.], [0., 0., 1.]])
    Ry = jnp.array([[ cp, 0., -sp], [0., 1., 0.], [sp, 0., cp]])
    Rx = jnp.array([[1., 0., 0.], [0., co, so], [0., -so, co]])
    R  = Rz @ Ry @ Rx

    local = (pts - cam[:3]) @ R.T          # (B, 3)
    z     = -local[:, 2]
    ok    = z > 0.0

    safe_z = jnp.where(ok, local[:, 2], -1.0)
    x = jnp.where(ok, -local[:, 0] / safe_z, 0.0)
    y = jnp.where(ok, -local[:, 1] / safe_z, 0.0)

    w2  = cal[0] / cal[2] / 2.0
    h2  = cal[1] / cal[2] / 2.0
    ins = ok & (x >= -w2) & (x < w2) & (y >= -h2) & (y < h2)
    y   = -y  # Agisoft y-flip

    r2  = x**2 + y**2
    dr  = (1.0
           + cal[5]*r2    + cal[6]*r2**2  + cal[7]*r2**3
           + cal[8]*r2**4 + cal[9]*r2**5)
    xp  = x*dr + cal[10]*(r2 + 2*x**2)   + 2*cal[11]*x*y*(1 + cal[12]*r2 + cal[13]*r2**2)
    yp  = y*dr + cal[11]*(r2 + 2*y**2)   + 2*cal[10]*x*y*(1 + cal[12]*r2 + cal[13]*r2**2)

    fx  = cal[0]*0.5 + cal[3] + xp*cal[2] + xp*cal[14] + yp*cal[15]
    fy  = cal[1]*0.5 + cal[4] + yp*cal[2]

    ins = ins & (fx >= 0) & (fx < cal[0]) & (fy >= 0) & (fy < cal[1])
    return fx, fy, z, ins


def _project_batched(pts_abs, cam_params, cal_jax, proj_batch_size):
    """
    Project pts_abs in fixed-size batches so _project_jax is compiled once.
    pts_abs    : (N, 3) float64 numpy
    cam_params : (6,)   float64 numpy
    cal_jax    : (16,)  float64 jax array
    """
    N   = len(pts_abs)
    pad = (-N) % proj_batch_size
    if pad:
        pts_jax = jnp.concatenate([jnp.array(pts_abs),
                                   jnp.zeros((pad, 3), dtype=jnp.float64)])
    else:
        pts_jax = jnp.array(pts_abs)
    cam_jax = jnp.array(cam_params)

    fx_l, fy_l, z_l, ins_l = [], [], [], []
    for s in range(0, N + pad, proj_batch_size):
        fx_b, fy_b, z_b, ins_b = _project_jax(pts_jax[s:s+proj_batch_size],
                                               cam_jax, cal_jax)
        fx_l.append(np.asarray(fx_b))
        fy_l.append(np.asarray(fy_b))
        z_l.append(np.asarray(z_b))
        ins_l.append(np.asarray(ins_b))

    return (np.concatenate(fx_l)[:N],
            np.concatenate(fy_l)[:N],
            np.concatenate(z_l)[:N],
            np.concatenate(ins_l)[:N])


def _build_global_depth_maps(xyz, meta, cal_jax, img_w, img_h, depth_buffer, proj_batch_size):
    """
    Build one depth map per camera using ALL zone points, streaming batch by batch
    so only proj_batch_size points are in JAX memory at a time and results are
    accumulated directly into the depth array (never materialising full-N arrays).
    Returns {cam_name: (img_h, img_w) float16} for cameras that see ≥1 point.
    The minimum_filter spread is already baked in so chunk-level calls can skip it.
    """
    N = len(xyz)
    depth_maps = {}

    for i, cam_name in enumerate(meta['names']):
        cam_jax = jnp.array(meta['cam_arrays'][i])
        depth   = np.full((img_h, img_w), np.inf, dtype=np.float32)
        has_any = False

        for s in range(0, N, proj_batch_size):
            batch = xyz[s : s + proj_batch_size]
            pad   = proj_batch_size - len(batch)
            if pad:
                pts_b = jnp.concatenate([jnp.array(batch),
                                         jnp.zeros((pad, 3), dtype=jnp.float64)])
            else:
                pts_b = jnp.array(batch)

            fx_b, fy_b, z_b, ins_b = _project_jax(pts_b, cam_jax, cal_jax)

            n      = len(batch)   # actual (non-padded) points in this batch
            ins_np = np.asarray(ins_b[:n])
            if not ins_np.any():
                continue
            has_any = True
            ux = np.asarray(fx_b[:n])[ins_np].astype(np.int32)
            uy = np.asarray(fy_b[:n])[ins_np].astype(np.int32)
            zv = np.asarray(z_b[:n])[ins_np].astype(np.float32)
            np.minimum.at(depth, (uy, ux), zv)

        if not has_any:
            continue
        if depth_buffer > 0:
            depth = minimum_filter(depth, size=2 * depth_buffer + 1,
                                   mode='constant', cval=np.inf)
        depth_maps[cam_name] = depth.astype(np.float16)

    return depth_maps


def _occlusion_mask(ux, uy, zv, img_w, img_h, depth_buffer, depth_thresh, global_depth=None):
    """
    Returns bool array (len = len(ux)). True = point is NOT occluded.
    When global_depth is provided (built from all zone points) it is used directly,
    which correctly handles occlusion across chunk boundaries.
    Falls back to a chunk-local depth buffer otherwise.
    """
    if global_depth is not None:
        return zv.astype(np.float32) <= global_depth[uy, ux].astype(np.float32) + depth_thresh

    x0 = max(int(ux.min()) - depth_buffer, 0)
    x1 = min(int(ux.max()) + depth_buffer + 1, img_w)
    y0 = max(int(uy.min()) - depth_buffer, 0)
    y1 = min(int(uy.max()) + depth_buffer + 1, img_h)

    depth  = np.full((y1 - y0, x1 - x0), np.inf, dtype=np.float32)
    ux_c   = ux - x0
    uy_c   = uy - y0
    np.minimum.at(depth, (uy_c, ux_c), zv.astype(np.float32))

    if depth_buffer > 0:
        depth = minimum_filter(depth, size=2 * depth_buffer + 1,
                               mode='constant', cval=np.inf)

    return zv.astype(np.float32) <= depth[uy_c, ux_c] + depth_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Camera selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_cameras(pts_abs, tile_center, cam_tree, meta, cal_jax,
                    img_w, img_h, target_h, target_w,
                    min_visible, n_candidates, max_cam,
                    depth_buffer, depth_thresh, proj_batch_size,
                    global_depth_maps=None):
    """
    Finds up to max_cam cameras that best cover the chunk.
    For each of the n_candidates nearest cameras (by XY position):
      1. Project all chunk points (JAX JIT, fixed-size batches).
      2. Apply depth-map occlusion on a cropped region.
      3. Discard if visible point count < min_visible (absolute).
    Cameras are ranked by quality = vis.sum() / N (fraction of the whole
    chunk that is unoccluded), so best-covering views are selected first.
    Returns list sorted by quality (best first), each dict:
        name  : str
        coord : (N, 2) int16   (u,v) in resized image; -1 = invisible
    """
    _, idxs = cam_tree.query(tile_center, k=min(n_candidates, len(meta['names'])))
    if np.ndim(idxs) == 0:
        idxs = [int(idxs)]

    N  = len(pts_abs)
    su = target_w / img_w
    sv = target_h / img_h

    selected = []
    diag     = []   # collected only when needed (printed if nothing is selected)

    for i in idxs:
        fx, fy, z, ins = _project_batched(pts_abs, meta['cam_arrays'][i],
                                          cal_jax, proj_batch_size)
        n_ins = int(ins.sum())
        if n_ins == 0:
            diag.append(f"  {meta['names'][i]}: 0/{N} in frustum")
            continue

        ux = fx[ins].astype(np.int32)
        uy = fy[ins].astype(np.int32)

        gd = global_depth_maps.get(meta['names'][i]) if global_depth_maps else None
        vis_sub = _occlusion_mask(ux, uy, z[ins], img_w, img_h,
                                  depth_buffer, depth_thresh, gd)
        vis = np.zeros(N, dtype=bool)
        vis[ins] = vis_sub

        n_vis   = int(vis.sum())
        quality = n_vis / N          # fraction of whole chunk visible
        diag.append(f"  {meta['names'][i]}: {n_ins} in frustum, "
                    f"{n_vis}/{N} visible after occlusion, quality={quality:.3f}"
                    f" (threshold={min_visible} pts)")

        if n_vis < min_visible:
            continue

        coord = np.full((N, 2), -1, dtype=np.int16)
        coord[vis, 0] = np.round(fx[vis] * su).astype(np.int16)
        coord[vis, 1] = np.round(fy[vis] * sv).astype(np.int16)

        selected.append({
            'name':    meta['names'][i],
            'quality': quality,
            'coord':   coord,
        })

    selected.sort(key=lambda c: c['quality'], reverse=True)
    return selected[:max_cam], diag


# ─────────────────────────────────────────────────────────────────────────────
# Image resizing
# ─────────────────────────────────────────────────────────────────────────────

def _resize_images(src_dir, dst_dir, target_h, target_w):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in sorted(src_dir.iterdir())
             if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]

    def _one(src):
        dst = dst_dir / src.name
        if not dst.exists():
            Image.open(src).convert('RGB') \
                 .resize((target_w, target_h), Image.Resampling.LANCZOS) \
                 .save(dst, format='JPEG', quality=95)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_one, paths))

    print(f"[images] {len(paths)} → {dst_dir}  ({target_w}×{target_h})")


# ─────────────────────────────────────────────────────────────────────────────
# Zone pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_mask(pts, max_half_extent=(30.0, 30.0, 150.0)):
    """Keep-mask dropping non-finite points and far outliers (per-axis
    ``median ± max_half_extent`` metres). Applied per chunk BEFORE orig_idx / xyz are
    saved so the reconstruction back-pointers stay aligned with the points the model
    sees, and so a stray point can't overflow spconv's int32 voxel index (silent CUDA
    crash). Mirrors datasets.utils.sanitize_points (same bound GridNet used at load
    time), moved here so it never desyncs predictions from orig_idx.
    """
    pts = np.asarray(pts)
    bound = np.asarray(max_half_extent, dtype=np.float64)
    keep = np.isfinite(pts).all(axis=1)
    valid = pts[keep]
    if valid.shape[0]:
        center = np.median(valid, axis=0)
        keep &= (np.abs(pts - center) <= bound).all(axis=1)
    return keep


def preprocess_zone(zone_dir, out_dir,
                    target_h, target_w,
                    chunk_size, stride, min_pts,
                    max_cam, min_visible, n_candidates,
                    depth_buffer, depth_thresh,
                    voxel_size, proj_batch_size,
                    n_workers,
                    save_orig_idx=False):
    zone_dir = Path(zone_dir)
    zone_out = Path(out_dir) / zone_dir.name

    # ── resize images ─────────────────────────────────────────────────────────
    _resize_images(zone_dir / 'images', zone_out / 'images', target_h, target_w)

    # ── load calibration + poses ──────────────────────────────────────────────
    cal      = _parse_calibration(zone_dir / 'pose' / 'camera_calibration.xml')
    cameras  = _read_cameras(zone_dir / 'pose' / 'cameras_pose.txt')
    img_w, img_h = cal['width'], cal['height']

    # Keep only cameras whose image actually exists — pose files sometimes
    # reference cameras not present in the images folder.
    available = {p.stem for p in (zone_dir / 'images').iterdir()
                 if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')}
    n_total  = len(cameras)
    cameras  = {k: v for k, v in cameras.items() if k in available}
    if not cameras:
        raise RuntimeError(f"[{zone_dir.name}] No cameras with matching images found.")
    if len(cameras) < n_total:
        print(f"[{zone_dir.name}] {n_total - len(cameras)} cameras in pose file have no image — skipped")

    names      = list(cameras.keys())
    centroids  = np.array([[c['Xs'], c['Ys']] for c in cameras.values()], dtype=np.float64)
    cam_arrays = np.array([[c['Xs'], c['Ys'], c['Zs'],
                            c['omega'], c['phi'], c['kappa']]
                           for c in cameras.values()], dtype=np.float64)
    meta     = {'names': names, 'centroids': centroids, 'cam_arrays': cam_arrays}
    cam_tree = cKDTree(centroids)   # built once, reused for all chunks
    cal_jax  = cal_to_jax(cal)      # packed calibration for JIT projection

    # ── Pass 1: stream LAS → write chunk point data → free LAS ───────────────
    print(f"[{zone_dir.name}] loading LAS ...")
    las_path = next((zone_dir / 'lidar').glob('*.las'))
    xyz, rgb, intensity, labels = _load_zone_points(las_path)
    print(f"[{zone_dir.name}] {len(xyz):,} pts  {len(names)} cams"
          f"{'  (no labels — test set)' if labels is None else ''}")

    # For test sets only: track which original LAS row each chunk point came from
    # so predictions can be reconstructed back to the original LAS point order.
    n_las_pts    = len(xyz) if save_orig_idx else None
    keep_indices = None
    if voxel_size > 0:
        xyz, rgb, intensity, labels, keep_indices = _voxel_downsample(
            xyz, rgb, intensity, labels, voxel_size)
        print(f"[{zone_dir.name}] after voxel({voxel_size}m): {len(xyz):,} pts")

    n_chunks = 0
    n_sanitized = 0
    for chunk in _iter_chunks(xyz, rgb, intensity, labels, chunk_size, stride, min_pts):
        # Sanitize BEFORE deriving anything (esp. orig_idx / xyz): drop non-finite +
        # far-outlier points so the reconstruction back-pointers stay aligned with the
        # points the model sees, and so a stray point can't overflow spconv's int32
        # voxel index. Filter every per-point array (incl. idx_in_input → orig_idx).
        keep = _sanitize_mask(chunk['pts'])
        if not keep.all():
            if int(keep.sum()) < min_pts:
                continue
            n_sanitized += int((~keep).sum())
            chunk['pts']          = chunk['pts'][keep]
            chunk['rgb']          = chunk['rgb'][keep]
            chunk['intensity']    = chunk['intensity'][keep]
            chunk['idx_in_input'] = chunk['idx_in_input'][keep]
            if chunk['labels'] is not None:
                chunk['labels']   = chunk['labels'][keep]

        cd = zone_out / f'chunk_{n_chunks:05d}'
        cd.mkdir(parents=True, exist_ok=True)

        # Save — int16 cm, range ±327m, precision 1cm
        xyz_local = (chunk['pts'] - chunk['origin']).astype(np.float32)
        np.save(cd / 'xyz.npy', (xyz_local * 100).round().astype(np.int16))

        np.save(cd / 'rgb.npy', chunk['rgb'].astype(np.uint8))
        np.save(cd / 'intensity.npy', (chunk['intensity'] * 255).round().astype(np.uint8))
        if chunk['labels'] is not None:
            np.save(cd / 'labels.npy', ID2TRAINID[chunk['labels']])

        meta_dict = {
            'origin':      chunk['origin'].tolist(),
            'tile_center': chunk['tile_center'].tolist(),
            'bbox':        list(chunk['bbox']),
            'cam_names':   [],
        }

        if save_orig_idx:
            # orig_idx.npy maps each chunk point back to its row in the original LAS,
            # enabling direct index-based reconstruction without KD-tree matching.
            idx_in_vox = chunk['idx_in_input']
            orig_idx   = keep_indices[idx_in_vox] if keep_indices is not None else idx_in_vox
            np.save(cd / 'orig_idx.npy', orig_idx.astype(np.int64))
            meta_dict['n_las_pts'] = n_las_pts

        with open(cd / 'meta.json', 'w') as f:
            json.dump(meta_dict, f)
        n_chunks += 1

    print(f"[{zone_dir.name}] building global depth maps ({len(names)} cameras) ...")
    global_depth_maps = _build_global_depth_maps(
        xyz, meta, cal_jax, img_w, img_h, depth_buffer, proj_batch_size)
    print(f"[{zone_dir.name}] depth maps ready for {len(global_depth_maps)}/{len(names)} cameras")

    del xyz, rgb, intensity, labels
    san = f" [sanitized {n_sanitized:,} outlier/non-finite pts]" if n_sanitized else ""
    print(f"[{zone_dir.name}] pass1 done: {n_chunks} chunks{san}, LAS freed")

    # ── Pass 2: project each chunk → add image_coord/image_mask → del abs ─────
    # Semaphore limits how many chunks are loaded at once.
    sem = Semaphore(max(n_workers * 2, 2))

    def _project_chunk(chunk_dir):
        try:
            with open(chunk_dir / 'meta.json') as f:
                m = json.load(f)
            origin  = np.array(m['origin'])
            pts_abs = np.load(chunk_dir / 'xyz.npy').astype(np.float64) * 0.01 + origin

            selected, diag = _select_cameras(
                pts_abs, np.array(m['tile_center']),
                cam_tree, meta, cal_jax,
                img_w, img_h, target_h, target_w,
                min_visible, n_candidates, max_cam,
                depth_buffer, depth_thresh, proj_batch_size,
                global_depth_maps,
            )
            if not selected:
                print(f"  [no camera] {chunk_dir.name}  N={len(pts_abs):,}"
                      f"  tile_center={m['tile_center']}")
                for line in diag:
                    print(line)
                return False

            N           = len(pts_abs)
            image_coord = np.full((N, max_cam, 2), -1, dtype=np.int16)
            cam_names   = [''] * max_cam
            for j, cam in enumerate(selected):
                image_coord[:, j] = cam['coord']
                cam_names[j]      = cam['name']

            # coord=-1 means invisible; visibility mask: coord[..., 0] >= 0
            np.savez_compressed(chunk_dir / 'image_coord.npz', coord=image_coord)
            m['cam_names'] = cam_names
            with open(chunk_dir / 'meta.json', 'w') as f:
                json.dump(m, f)
            return True
        finally:
            sem.release()

    chunk_dirs = sorted(zone_out.glob('chunk_*'))
    saved = skipped = total = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = []
        for cd in chunk_dirs:
            sem.acquire()
            futs.append(pool.submit(_project_chunk, cd))
        for fut in as_completed(futs):
            total += 1
            if fut.result():
                saved += 1
            else:
                skipped += 1
            if total % 50 == 0:
                print(f"  [{zone_dir.name}] {total}/{n_chunks} projected")

    print(f"[{zone_dir.name}] pass2 done — {saved} saved, {skipped} skipped (no cameras)")


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(out_root, out_path):
    """Scans out_root/{train,val,test}/{zone}/chunk_* and writes manifest.json."""
    out_root = Path(out_root)
    manifest = {'train': [], 'val': [], 'test': []}
    image_free = {'train': 0, 'val': 0, 'test': 0}   # chunks with no image_coord.npz
    for split in ('train', 'val', 'test'):
        split_dir = out_root / split
        if not split_dir.exists():
            print(f"  {split_dir} does not exist — skipping")
            continue
        for zone_dir in sorted(split_dir.iterdir()):
            if not zone_dir.is_dir():
                continue
            for chunk_dir in sorted(zone_dir.glob('chunk_*')):
                # Include every chunk that has point data, regardless of camera
                # coverage. Chunks without image_coord.npz (no cameras) are still
                # trained/evaluated as image-free — the dataset + model handle the
                # zero-camera case (missing-feature embedding). Filtering on images
                # here would silently drop those chunks from train/val/test and
                # leave their points uncovered at reconstruction time.
                if (chunk_dir / 'xyz.npy').exists():
                    manifest[split].append({
                        'zone':      zone_dir.name,
                        'chunk_dir': str(chunk_dir.relative_to(out_root)),
                    })
                    if not (chunk_dir / 'image_coord.npz').exists():
                        image_free[split] += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    for split, entries in manifest.items():
        n = len(entries)
        n_free = image_free[split]
        print(f"  {split}: {n} chunks ({n_free} image-free, {n - n_free} with images)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_all(raw_root, out_root, split_json, split,
                   target_h, target_w,
                   chunk_size=20.0, stride=10.0, min_pts=512,
                   max_cam=10, min_visible=100, n_candidates=40,
                   depth_buffer=3, depth_thresh=0.5,
                   voxel_size=0.0, proj_batch_size=65536,
                   n_workers=4):
    """
    Processes all zones for a given split, one zone at a time
    (only one LAS file in memory at once).
    """
    raw_root = Path(raw_root)
    with open(split_json) as f:
        splits = json.load(f)

    zone_names = splits.get(split, [])
    if not zone_names:
        print(f"No zones for split '{split}' in {split_json}")
        return

    zones = [raw_root / z for z in zone_names if (raw_root / z).is_dir()]
    print(f"Split '{split}': {len(zones)} zones")

    split_out = Path(out_root) / split
    
    # i=0
    for zone_dir in zones:
        preprocess_zone(
            zone_dir, split_out,
            target_h=target_h, target_w=target_w,
            chunk_size=chunk_size, stride=stride, min_pts=min_pts,
            max_cam=max_cam, min_visible=min_visible, n_candidates=n_candidates,
            depth_buffer=depth_buffer, depth_thresh=depth_thresh,
            voxel_size=voxel_size, proj_batch_size=proj_batch_size,
            n_workers=n_workers,
            save_orig_idx=(split == "test"),
        )
        # i+=1
        # if i>1:
        #     break
        

    build_manifest(out_root, Path(out_root) / 'manifest.json')
    print(f"Split '{split}' complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GridNet-HD preprocessing')
    parser.add_argument('--raw-root',     required=True)
    parser.add_argument('--out-root',     required=True)
    parser.add_argument('--split-json',   required=True)
    parser.add_argument('--split',        required=True,  help='train | val | test')
    parser.add_argument('--target-h',     type=int, required=True)
    parser.add_argument('--target-w',     type=int, required=True)
    parser.add_argument('--chunk-size',   type=float, default=20.0)
    parser.add_argument('--stride',        type=float, default=10.0,
                        help='Step between windows (< chunk-size gives overlap)')
    parser.add_argument('--min-pts',      type=int,   default=10000)
    parser.add_argument('--max-cam',      type=int,   default=10)
    parser.add_argument('--min-visible',  type=int,   default=100,
                        help='Min visible points (absolute) for a camera to be kept')
    parser.add_argument('--n-candidates', type=int,   default=40)
    parser.add_argument('--depth-buffer', type=int,   default=3)
    parser.add_argument('--depth-thresh', type=float, default=0.5)
    parser.add_argument('--voxel-size',      type=float, default=0.0,
                        help='Voxel size for downsampling (0 = disabled)')
    parser.add_argument('--proj-batch-size', type=int,   default=65536,
                        help='Points per JAX JIT batch for projection (avoids recompilation)')
    parser.add_argument('--workers',         type=int,   default=1)
    args = parser.parse_args()

    preprocess_all(
        raw_root=args.raw_root,
        out_root=args.out_root,
        split_json=args.split_json,
        split=args.split,
        target_h=args.target_h,
        target_w=args.target_w,
        chunk_size=args.chunk_size,
        stride=args.stride,
        min_pts=args.min_pts,
        max_cam=args.max_cam,
        min_visible=args.min_visible,
        n_candidates=args.n_candidates,
        depth_buffer=args.depth_buffer,
        depth_thresh=args.depth_thresh,
        voxel_size=args.voxel_size,
        proj_batch_size=args.proj_batch_size,
        n_workers=args.workers,
    )
