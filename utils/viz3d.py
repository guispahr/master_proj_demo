"""
utils/viz3d.py — general 3D point-cloud visualization helpers (dataset-agnostic)
================================================================================

Rendering / export / coloring utilities shared by the dataset-specific viz scripts
(e.g. datasets/preprocessing/helimap/visualize_predictions.py and render_video.py).

Depends only on numpy + open3d (render / show / ply) + laspy (las) + cv2 (video) —
no torch, so the standalone scripts stay lightweight. Everything here is generic:
it takes plain (coords, colors) arrays plus a class palette, so a new dataset only
needs its own loader; the drawing/exporting code is reused unchanged.

Conventions
-----------
- coords : (N, 3) float, real metric coordinates (any frame; centered internally).
- colors : (N, 3) float in [0, 1].
- palette: (K, 3) float in [0, 1], one row per class id.
- Camera: an "oblique" view looks from ``direction`` (eye offset from the look-at
  point) with world up ``up``; renders on a pure white background.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None
try:
    import laspy
except ImportError:
    laspy = None


def _require_o3d():
    if o3d is None:
        raise ImportError("open3d is required for this operation: pip install open3d")


def _require_laspy():
    if laspy is None:
        raise ImportError("laspy is required for LAS export: pip install laspy")


# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────

GREY = np.array([0.45, 0.45, 0.45], np.float32)
GREEN = np.array([0.10, 0.80, 0.10], np.float32)
RED = np.array([0.90, 0.10, 0.10], np.float32)
ORANGE = np.array([1.00, 0.55, 0.00], np.float32)
FAINT = np.array([0.72, 0.72, 0.72], np.float32)


def to_rgb01(color) -> np.ndarray:
    """A '#rrggbb' hex string or an (R, G, B) tuple in 0-255 → float RGB in [0, 1]."""
    if isinstance(color, str):
        h = color.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
    return np.asarray(color[:3], dtype=np.float32) / 255.0


def class_colors(ids: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map integer class ids → (N, 3) RGB via ``palette``; out-of-range ids → grey."""
    ids = np.asarray(ids)
    out = np.tile(GREY, (len(ids), 1)).astype(np.float32)
    valid = (ids >= 0) & (ids < len(palette))
    out[valid] = palette[ids[valid]]
    return out


def colorize(mode, gt, pred, palette, ignore_index, confusion=None, rgb=None):
    """Per-point colors for a visualization ``mode`` → (colors (N,3) float, correct (N,)).

    ``correct``: 1 correct, 0 wrong, 2 ignored/unknown (kept for LAS scalar fields).

    Modes: ``pred`` / ``gt`` (class palette), ``rgb`` (true point colors), ``error``
    (green/red/grey), ``confusion`` (isolate one gt→pred mistake given
    ``confusion=(gt_id, pred_id)``).
    """
    n = len(pred)
    if gt is None:
        correct = np.full(n, 2, np.uint8)
    else:
        valid = gt != ignore_index
        correct = np.where(valid, (pred == gt).astype(np.uint8), 2).astype(np.uint8)

    if mode == "pred":
        return class_colors(pred, palette), correct
    if mode == "gt":
        if gt is None:
            raise ValueError("color-by gt needs labels (none found for this split)")
        return class_colors(gt, palette), correct
    if mode == "rgb":
        if rgb is None:
            raise ValueError("color-by rgb needs rgb.npy in the chunks (none found)")
        return np.clip(rgb, 0, 1).astype(np.float32), correct
    if mode == "error":
        if gt is None:
            raise ValueError("color-by error needs labels (none found for this split)")
        colors = np.tile(GREY, (n, 1)).astype(np.float32)
        colors[correct == 1] = GREEN
        colors[correct == 0] = RED
        return colors, correct
    if mode == "confusion":
        if gt is None:
            raise ValueError("color-by confusion needs labels (none found for this split)")
        a, b = confusion
        colors = np.tile(FAINT, (n, 1)).astype(np.float32)
        is_a = gt == a
        colors[is_a & (pred == a)] = GREEN       # class A, correct
        colors[is_a & (pred != a)] = ORANGE      # class A, any error
        colors[is_a & (pred == b)] = RED         # class A predicted as B
        return colors, correct
    raise ValueError(f"unknown color mode {mode!r}")


def voxel_downsample_idx(coords, voxel):
    """Keep-one-per-voxel indices for display downsampling, or None if ``voxel<=0``."""
    if voxel is None or voxel <= 0:
        return None
    vox = np.floor(np.asarray(coords) / voxel).astype(np.int64)
    vox -= vox.min(0)
    mx = vox.max(0) + 1
    keys = vox[:, 0] * (mx[1] * mx[2]) + vox[:, 1] * mx[2] + vox[:, 2]
    _, keep = np.unique(keys, return_index=True)
    keep.sort()
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Camera / Open3D scene
# ─────────────────────────────────────────────────────────────────────────────

def _camera_basis(direction, up):
    direction = np.asarray(direction, np.float64); direction /= np.linalg.norm(direction)
    up = np.asarray(up, np.float64); up /= np.linalg.norm(up)
    forward = -direction
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return direction, up, right, cam_up


def fit_camera(P, direction, up, fov, aspect, margin=1.03):
    """Fit a still camera to the *projected silhouette* of centered points ``P``.

    Returns (look_at, eye, up) float32; the scene fills the frame with balanced margin.
    """
    direction, up, right, cam_up = _camera_basis(direction, up)
    x, y = P @ right, P @ cam_up
    x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    half_w = max(0.5 * (x1 - x0), 1e-3)
    half_h = max(0.5 * (y1 - y0), 1e-3)
    vfov = np.radians(fov)
    hfov = 2.0 * np.arctan(np.tan(vfov / 2) * aspect)
    dist = max(half_h / np.tan(vfov / 2), half_w / np.tan(hfov / 2)) * margin
    look_at = 0.5 * (x0 + x1) * right + 0.5 * (y0 + y1) * cam_up
    eye = look_at + direction * dist
    return look_at.astype(np.float32), eye.astype(np.float32), up.astype(np.float32)


def _make_pcd(P, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(P, np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    return pcd


def _make_renderer(width, height, point_size):
    """OffscreenRenderer with a pure white background (tone mapping disabled so the
    white stays white and unlit colors render exactly)."""
    r = o3d.visualization.rendering.OffscreenRenderer(width, height)
    try:
        r.scene.view.set_post_processing(False)
    except Exception:
        pass
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader, mat.point_size = "defaultUnlit", float(point_size)
    return r, mat


# ─────────────────────────────────────────────────────────────────────────────
# Still image / interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

def _content_bbox(img, pad=8, bg_thresh=250):
    """(y0, y1, x0, x1) bounding box of the non-white content in an (H,W,3) image, + pad."""
    mask = np.asarray(img).min(axis=2) < bg_thresh
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    h, w = img.shape[:2]
    return (max(0, rows[0] - pad), min(h, rows[-1] + 1 + pad),
            max(0, cols[0] - pad), min(w, cols[-1] + 1 + pad))


def _crop_to_content(img, pad=8):
    """Crop an image to the bounding box of its non-white pixels (+ pad)."""
    y0, y1, x0, x1 = _content_bbox(img, pad)
    return np.ascontiguousarray(np.asarray(img)[y0:y1, x0:x1])


def _legend_panel(legend, height, bg=(255, 255, 255)):
    """A white legend panel (numpy RGB, ``height`` tall): a color swatch + label per
    entry, vertically centered. ``legend`` = list of ``(label, (r, g, b) in 0-255)``.
    Text is drawn crisply at the final resolution (no anti-alias blur)."""
    from PIL import Image, ImageDraw, ImageFont
    fs = max(12, int(height / 42))                  # font size scales with the image
    sw, pad_, gap = int(fs * 1.25), int(fs * 0.9), int(fs * 0.5)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fs)
    except Exception:
        try:
            import matplotlib
            font = ImageFont.truetype(
                str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), fs)
        except Exception:
            font = ImageFont.load_default()
    meas = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    label_w = max((meas.textlength(str(l), font=font) for l, _ in legend), default=0)
    panel = Image.new("RGB", (int(pad_ + sw + gap + label_w + pad_), height), bg)
    draw = ImageDraw.Draw(panel)
    row = sw + gap
    y = max(pad_, (height - (row * len(legend) - gap)) // 2)   # vertically centered
    for label, col in legend:
        col = tuple(int(c) for c in col[:3])
        draw.rectangle([pad_, y, pad_ + sw, y + sw], fill=col, outline=(60, 60, 60))
        draw.text((pad_ + sw + gap, y + sw / 2), str(label), fill=(20, 20, 20),
                  font=font, anchor="lm")
        y += row
    return np.asarray(panel)


def render_png(out_path, coords, colors, width=2560, height=1440, fov=25.0,
               point_size=2.5, margin=1.03, direction=(0.6, -0.6, 0.5), up=(0, 0, 1),
               crop=False, pad=8, ssaa=1, legend=None):
    """Render a white-background PNG framed tightly on the point cloud.

    ``crop`` removes the white margins the camera fit can't (aspect ratio mismatch +
    perspective foreshortening) by cropping to the non-white content + ``pad`` px.

    ``ssaa`` = super-sampling anti-aliasing: render at ``ssaa``× resolution then
    area-downscale to ``width``×``height``. This is the main "make it sharper / less
    blurry" knob — it smooths point edges and removes aliasing. ``2`` is a good default;
    ``3–4`` for print-quality figures (slower). Point size scales with it automatically.
    """
    _require_o3d()
    coords = np.asarray(coords, np.float64)
    P = coords - 0.5 * (coords.min(0) + coords.max(0))
    ssaa = max(1, int(ssaa))
    sw, sh = width * ssaa, height * ssaa
    r, mat = _make_renderer(sw, sh, point_size * ssaa)
    r.scene.add_geometry("pcd", _make_pcd(P, colors), mat)
    look_at, eye, up_v = fit_camera(P, direction, up, fov, sw / sh, margin)
    r.setup_camera(fov, look_at, eye, up_v)
    arr = np.asarray(r.render_to_image())
    if ssaa != 1:
        import cv2
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    if crop:
        arr = _crop_to_content(arr, pad)
    if legend:                                     # class-color key on the right
        arr = np.hstack([arr, _legend_panel(legend, arr.shape[0])])
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out_path), o3d.geometry.Image(np.ascontiguousarray(arr)))
    print(f"  saved PNG → {out_path}  ({arr.shape[1]}×{arr.shape[0]})")


def show_interactive(coords, colors, title="", up=(0, 0, 1)):
    """Open an interactive Open3D window (writes nothing). Blocks until closed (Q).
    Needs a display: on WSL run `export DISPLAY=:0`; headless nodes have none."""
    _require_o3d()
    coords = np.asarray(coords, np.float64)
    pcd = _make_pcd(coords - coords.mean(0), colors)
    extent = float(np.ptp(coords, axis=0).max()) if len(coords) else 1.0
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.05 * extent, 0.5))
    print(f"  opening viewer: {title} — press Q to close")
    o3d.visualization.draw_geometries([pcd, frame], window_name=title or "point cloud")


# ─────────────────────────────────────────────────────────────────────────────
# 360° orbit video
# ─────────────────────────────────────────────────────────────────────────────

def render_orbit_video(out_path, coords, colors, n_frames=180, fps=30, elevation=30.0,
                       fov=25.0, width=1920, height=1080, point_size=2.5, margin=1.1,
                       up=(0, 0, 1), gif_scale=0.5, crop=False, pad=8, ssaa=1, legend=None):
    """Render a 360° turntable orbiting the cloud about its vertical axis.

    Output format is picked from ``out_path``'s extension: ``.gif`` → an animated GIF
    (frames downscaled by ``gif_scale`` and encoded with Pillow — good for an
    autoplaying README banner, no ffmpeg needed); anything else → mp4 (cv2). The camera
    distance is fixed for the whole spin (so the object doesn't pulse in size): sized to
    the horizontal bounding circle + vertical extent at ``elevation`` so nothing clips.

    ``crop`` trims white margins by cropping every frame to a *common* bounding box (the
    union of non-white content over the whole spin, + ``pad`` px) so the scene fills the
    frame with no white borders while the frame size stays constant.
    """
    _require_o3d()
    import cv2
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    is_gif = out_path.suffix.lower() == ".gif"

    coords = np.asarray(coords, np.float64)
    P = coords - 0.5 * (coords.min(0) + coords.max(0))
    ssaa = max(1, int(ssaa))                       # super-sample then downscale → anti-aliasing
    sw, sh = width * ssaa, height * ssaa
    r, mat = _make_renderer(sw, sh, point_size * ssaa)
    r.scene.add_geometry("pcd", _make_pcd(P, colors), mat)

    up_v = np.asarray(up, np.float64); up_v /= np.linalg.norm(up_v)
    ext = P.max(0) - P.min(0)
    R_xy = 0.5 * float(np.hypot(ext[0], ext[1]))   # horizontal radius (const while spinning)
    half_z = 0.5 * float(ext[2])
    el = np.radians(float(np.clip(elevation, 1.0, 89.0)))
    half_h = half_z * np.cos(el) + R_xy * np.sin(el)   # worst-case vertical screen extent
    half_w = max(R_xy, 1e-3)
    vfov = np.radians(fov)
    hfov = 2.0 * np.arctan(np.tan(vfov / 2) * (width / height))
    dist = max(half_h / np.tan(vfov / 2), half_w / np.tan(hfov / 2)) * margin

    # Post-processing (union-crop, GIF encode, or legend) needs every frame → buffer;
    # a plain mp4 without any of those streams frame-by-frame to keep memory flat.
    buffer = is_gif or crop or (legend is not None)
    frames = []
    vw = None
    if not buffer:
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not vw.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {out_path} (codec/ffmpeg issue)")

    look_at = np.zeros(3, np.float32)
    step = max(1, n_frames // 10)
    print(f"  rendering {n_frames} frames ({'gif' if is_gif else 'mp4'}"
          f"{', crop' if crop else ''})…", flush=True)
    for i in range(n_frames):
        az = 2.0 * np.pi * i / n_frames
        d = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        eye = (d * dist).astype(np.float32)
        r.setup_camera(fov, look_at, eye, up_v.astype(np.float32))
        frame = np.asarray(r.render_to_image())          # (sh, sw, 3) RGB uint8
        if ssaa != 1:                                    # AA downscale to target size
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if buffer:
            frames.append(frame)
        else:
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % step == 0:
            print(f"    frame {i + 1}/{n_frames}", flush=True)

    if not buffer:                                   # streamed mp4, no crop
        vw.release()
        print(f"  saved video → {out_path}  ({n_frames} frames @ {fps}fps, {width}×{height})")
        return

    if crop:                                         # one crop box for the whole spin
        y0 = x0 = 1 << 30; y1 = x1 = 0
        for f in frames:
            a, b, c, d2 = _content_bbox(f, pad)
            y0, y1, x0, x1 = min(y0, a), max(y1, b), min(x0, c), max(x1, d2)
        if (y1 - y0) % 2:                            # even dims keep video codecs happy
            y1 -= 1
        if (x1 - x0) % 2:
            x1 -= 1
        frames = [np.ascontiguousarray(f[y0:y1, x0:x1]) for f in frames]

    if is_gif and gif_scale != 1.0:                  # shrink for a small GIF (before legend)
        fh, fw = frames[0].shape[:2]
        size = (max(1, int(fw * gif_scale)), max(1, int(fh * gif_scale)))
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]

    if legend:                                       # same class-color key on every frame
        panel = _legend_panel(legend, frames[0].shape[0])
        frames = [np.ascontiguousarray(np.hstack([f, panel])) for f in frames]

    fh, fw = frames[0].shape[:2]
    if is_gif:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(out_path), save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(fps, 1)), loop=0, optimize=True, disposal=2)
        print(f"  saved gif → {out_path}  ({n_frames} frames @ {fps}fps, {fw}×{fh})")
    else:                                            # buffered mp4 (crop / legend path)
        if fw % 2:                                   # even width keeps codecs happy
            frames = [f[:, :-1] for f in frames]; fw -= 1
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        if not vw.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {out_path} (codec/ffmpeg issue)")
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"  saved video → {out_path}  ({n_frames} frames @ {fps}fps, {fw}×{fh})")


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def write_las(out_path, coords, colors, extra_fields=None):
    """Write a LAS (point_format 3: XYZ + RGB). ``extra_fields`` is an optional dict
    of {name: uint8 array} added as per-point scalar dimensions (e.g. pred/gt/correct),
    so a viewer like CloudCompare can recolor or filter by them."""
    _require_laspy()
    coords = np.asarray(coords, np.float64)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = coords.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    c16 = (np.clip(colors, 0, 1) * 65535).astype(np.uint16)
    las.red, las.green, las.blue = c16[:, 0], c16[:, 1], c16[:, 2]
    for name, arr in (extra_fields or {}).items():
        if arr is None:
            continue
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8))
        setattr(las, name, np.clip(np.asarray(arr), 0, 255).astype(np.uint8))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    print(f"  saved LAS → {out_path}  ({len(coords):,} pts)")


def write_ply(out_path, coords, colors):
    _require_o3d()
    coords = np.asarray(coords, np.float64)
    pcd = _make_pcd(coords - coords.mean(0), colors)   # center: float32 precision
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"  saved PLY → {out_path}  ({len(coords):,} pts)")
