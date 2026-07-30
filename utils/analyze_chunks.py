"""
utils/analyze_chunks.py
========================

Unified statistics + plots for preprocessed chunk datasets:
GridNet (GridNet-HD), the two Helimap datasets (Atocha, A9CR), and nuScenes.
Usage:
    python -m utils.analyze_chunks --dataset atocha  --path /data/preprocessed_ATOCHA \\
        --splits train val test --out-dir analysis/atocha

    python -m utils.analyze_chunks --dataset gridnet --path /data/GridNet/preprocessed \\
        --splits train --out-dir analysis/gridnet --no-images

    python -m utils.analyze_chunks --dataset nuscenes --path /rcp/gspahr/nuscenes_preprocessed \\
        --splits train val --out-dir analysis/nuscenes --no-images
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _rgb_to_hex(c) -> str:
    """RGB tuple in [0, 255] (or a hex string) → '#rrggbb'."""
    if isinstance(c, str):
        return c
    r, g, b = (int(round(v)) for v in c[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _colors_to_mpl(label_colors, n):
    """n matplotlib RGB-float colors aligned to classes, from a dataset's
    CLASS_COLORS (RGB 0-255 tuples or hex strings)
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    out = []
    for i in range(n):
        if label_colors and i < len(label_colors):
            c = label_colors[i]
            out.append(c if isinstance(c, str) else tuple(v / 255.0 for v in c[:3]))
        else:
            out.append(cmap(i % 20))
    return out


def _analysis_collate(batch):
    """batch_size=1 passthrough that drops the decoded camera pixels.

    The analysis only consumes coord / segment / image_coord / image_mask, never
    the `image` pixel arrays.
    """
    sample = batch[0]
    if isinstance(sample, dict):
        sample.pop("image", None)
        # nuScenes also returns a per-point (N, CAM) camera-depth array the
        # analysis never reads; drop it so it isn't pickled back to the main
        # process (comparable in size to the pixels for dense frames).
        sample.pop("image_depth", None)
    return sample


# ignore_index used by each dataset's label space (see configs/<name>/base.yaml)
_IGNORE_INDEX = {
    "gridnet": 11,    # "unlabeled" is the last of the 12 GridNet classes
    "atocha":  255,
    "a9cr":    255,
    "nuscenes": -1,   # learning_map sends unlabeled/rare raw classes to -1
}

_NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface",
    "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]
_NUSCENES_CLASS_COLORS = [
    (255, 120,  50),  # 0  barrier
    (255, 192, 203),  # 1  bicycle
    (255, 255,   0),  # 2  bus
    (  0, 150, 245),  # 3  car
    (  0, 255, 255),  # 4  construction_vehicle
    (255, 127,   0),  # 5  motorcycle
    (255,   0,   0),  # 6  pedestrian
    (255, 240, 150),  # 7  traffic_cone
    (135,  60,   0),  # 8  trailer
    (160,  32, 240),  # 9  truck
    (255,   0, 255),  # 10 driveable_surface
    (139, 137, 137),  # 11 other_flat
    ( 75,   0,  75),  # 12 sidewalk
    (150, 240,  80),  # 13 terrain
    (230, 230, 250),  # 14 manmade
    (  0, 175,   0),  # 15 vegetation
]

_FALLBACK_CLASS_NAMES = {"nuscenes": _NUSCENES_CLASS_NAMES}
_FALLBACK_CLASS_COLORS = {"nuscenes": _NUSCENES_CLASS_COLORS}


def _build_dataset(dataset_name: str, path: str, split: str, with_image: bool, sweeps: int = 10):
    name = dataset_name.lower()
    if name == "gridnet":
        from datasets import GridNetPairDataset
        return GridNetPairDataset(path, split, transform=None, with_normals=False, with_image=with_image)
    elif name == "atocha":
        from datasets import AtochaPairDataset
        return AtochaPairDataset(path, split, transform=None, with_image=with_image)
    elif name == "a9cr":
        from datasets import A9CRPairDataset
        return A9CRPairDataset(path, split, transform=None, with_image=with_image)
    elif name == "nuscenes":
        from datasets import NuScenesDataset
        return NuScenesDataset(
            data_root=path, split=split, transform=None,
            with_images=with_image, sweeps=sweeps,
        )
    raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: gridnet, atocha, a9cr, nuscenes")

def compute_stats(dataset, label_names=None, label_colors=None, ignore_index=None,
                  coverage_thresholds=(0.10, 0.25, 0.50), num_workers=0):
    """
    One pass over `dataset`, computing both chunk-level and coverage stats.

    Returns (chunk_stats, coverage_stats) dicts — see `_format_chunk_stats`
    and `_format_coverage_stats` for their contents.
    """
    # chunk accumulators
    n_points_list = []
    extents = []
    n_classes = len(label_names) if label_names else 0
    label_counts = np.zeros(n_classes, dtype=np.int64)
    n_ignored = 0
    has_labels = True

    # coverage accumulators
    total_points = 0
    covered_points = 0
    per_cam_covered = None
    cov_hist = None
    per_sample_rates = []
    n_no_image_key = 0   # chunk carried no image_mask/image_coord at all (with_image off)
    n_image_free = 0     # chunk has the image structure but zero cameras project onto it
    C = 0

    iterator = dataset
    if num_workers and num_workers > 0:
        from torch.utils.data import DataLoader
        iterator = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=_analysis_collate,
            prefetch_factor=2,
        )
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(dataset),
                        desc="  analyzing chunks", unit="chunk")
    except ImportError:
        pass

    for sample in iterator:
        # ---- points / extent -------------------------------------------------
        coord = sample.get("coord")
        if isinstance(coord, torch.Tensor):
            coord = coord.numpy()
        N = int(coord.shape[0]) if coord is not None else 0
        n_points_list.append(N)
        if coord is not None and N > 0:
            extents.append(coord.max(axis=0) - coord.min(axis=0))

        # ---- labels -------------------------------------------------------
        segment = sample.get("segment")
        if segment is None:
            has_labels = False
        else:
            seg = segment.numpy() if isinstance(segment, torch.Tensor) else np.asarray(segment)
            seg = seg.reshape(-1).astype(np.int64)
            if ignore_index is not None:
                ignore_mask = seg == ignore_index
                n_ignored += int(ignore_mask.sum())
                seg = seg[~ignore_mask]
            if len(seg):
                max_id = int(seg.max()) + 1
                if max_id > len(label_counts):
                    grown = np.zeros(max_id, dtype=np.int64)
                    grown[: len(label_counts)] = label_counts
                    label_counts = grown
                unique, counts = np.unique(seg, return_counts=True)
                for u, c in zip(unique, counts):
                    label_counts[int(u)] += c

        # ---- image coverage -------------------------------------------------
        mask = None
        if "image_mask" in sample:
            mask = sample["image_mask"]
        elif "image_coord" in sample:
            mask = sample["image_coord"][..., 0] >= 0

        if mask is None:
            n_no_image_key += 1
            per_sample_rates.append(np.nan)
            continue

        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        mask = mask.astype(bool)

        n, c = mask.shape
        if c == 0:
            n_image_free += 1
            total_points += n
            if cov_hist is None:
                cov_hist = np.zeros(1, dtype=np.int64)
            cov_hist[0] += n
            # Excluded from the per-chunk coverage-rate distribution (reported
            # separately as n_chunks_image_free). Still counted as uncovered points.
            per_sample_rates.append(np.nan)
            continue
        if c > C:
            C = c
            per_cam_covered = (np.zeros(C, dtype=np.int64) if per_cam_covered is None
                               else np.pad(per_cam_covered, (0, C - len(per_cam_covered))))
            cov_hist = (np.zeros(C + 1, dtype=np.int64) if cov_hist is None
                        else np.pad(cov_hist, (0, C + 1 - len(cov_hist))))
        elif per_cam_covered is None:
            per_cam_covered = np.zeros(C, dtype=np.int64)
            cov_hist = np.zeros(C + 1, dtype=np.int64)

        per_point = mask.sum(axis=1)
        n_covered = int((per_point > 0).sum())

        total_points += n
        covered_points += n_covered
        per_cam_covered[:c] += mask.sum(axis=0).astype(np.int64)
        for k in range(C + 1):
            cov_hist[k] += int((per_point == k).sum())

        per_sample_rates.append(n_covered / n if n else np.nan)

    # assemble chunk_stats
    n_points = np.array(n_points_list, dtype=np.int64)
    extents_arr = np.array(extents, dtype=np.float64) if extents else np.zeros((0, 3))

    chunk_stats = dict(
        n_chunks=len(n_points),
        n_points=n_points,
        extents=extents_arr,
        has_labels=has_labels,
        label_counts=label_counts,
        label_names=label_names,
        label_colors=label_colors,
        ignore_index=ignore_index,
        n_ignored_points=n_ignored,
    )

    # assemble coverage_stats
    per_sample_rates = np.array(per_sample_rates, dtype=np.float64)
    valid = per_sample_rates[~np.isnan(per_sample_rates)]
    chunks_below = {t: int((valid < t).sum()) for t in coverage_thresholds}

    if per_cam_covered is None:
        per_cam_covered = np.zeros(0, dtype=np.int64)
        cov_hist = np.zeros(1, dtype=np.int64)

    coverage_stats = dict(
        n_chunks=len(per_sample_rates),
        n_chunks_no_image_key=n_no_image_key,
        n_chunks_image_free=n_image_free,
        total_points=total_points,
        covered_points=covered_points,
        uncovered_points=total_points - covered_points,
        coverage_rate=covered_points / total_points if total_points else 0.0,
        per_cam_covered=per_cam_covered,
        per_cam_rate=per_cam_covered / total_points if total_points else np.zeros(C),
        coverage_histogram=cov_hist,
        per_sample_rates=per_sample_rates,
        chunks_below=chunks_below,
    )

    return chunk_stats, coverage_stats

def _format_chunk_stats(stats: dict, label: str = "") -> str:
    lines = []
    sep = "─" * 62
    title = f"Chunk statistics{' — ' + label if label else ''}"
    lines += [f"\n{sep}", title, sep]

    n = stats["n_chunks"]
    pts = stats["n_points"]
    lines.append(f"  Chunks examined      : {n:>10,}")

    if n > 0:
        lines += [
            "\n  Points per chunk:",
            f"    min    = {pts.min():>12,}",
            f"    max    = {pts.max():>12,}",
            f"    mean   = {pts.mean():>12,.1f}",
            f"    median = {np.median(pts):>12,.1f}",
            f"    std    = {pts.std():>12,.1f}",
            f"    total  = {pts.sum():>12,}",
        ]

    ext = stats.get("extents")
    if ext is not None and len(ext) > 0:
        lines.append("\n  Spatial extent per chunk (m):")
        for i, ax in enumerate(("X", "Y", "Z")):
            e = ext[:, i]
            lines.append(
                f"    {ax}:  min={e.min():7.2f}  max={e.max():7.2f}"
                f"  mean={e.mean():7.2f}  median={np.median(e):7.2f}"
            )

    if stats["has_labels"]:
        lc = stats["label_counts"]
        names = stats.get("label_names") or []
        ignore = stats.get("ignore_index")
        n_ignored = stats.get("n_ignored_points", 0)
        total_valid = int(lc.sum())
        total = total_valid + n_ignored

        header = f"  Label distribution (scored: {total_valid:,}"
        if ignore is not None:
            header += f", ignored [{ignore}]: {n_ignored:,}"
        lines.append(f"\n{header}):")

        colors = stats.get("label_colors") or []
        for i, count in enumerate(lc):
            name = names[i] if i < len(names) else f"class {i}"
            hexc = _rgb_to_hex(colors[i]) if i < len(colors) else "   -   "
            pct = count / total_valid * 100 if total_valid else 0.0
            bar = "█" * int(count / max(lc.max(), 1) * 40)
            lines.append(f"    {i:3d} {name:20s} {hexc:>7s}: {count:>10,}  ({pct:5.1f} %)  {bar}")

        if ignore is not None and n_ignored:
            pct = n_ignored / total * 100 if total else 0.0
            lines.append(f"    {'':3} {'[ignored]':20s}: {n_ignored:>10,}  ({pct:5.1f} % of all points)")

    lines.append(sep)
    return "\n".join(lines)


def print_chunk_stats(stats: dict, label: str = "") -> None:
    print(_format_chunk_stats(stats, label))


def _save_chunk_stats_figure(stats: dict, fig_path, label: str = "") -> None:
    """Two-row dashboard: percentages on top, absolute counts on the bottom.

    Columns: chunk-size distribution, label distribution (bars colored by each
    class's CLASS_COLORS so the figure matches a 3D segmentation render), and the
    per-axis spatial extent (count-only → spans both rows).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig_path = Path(fig_path)
    pts = stats["n_points"]
    has_ext = stats.get("extents") is not None and len(stats["extents"]) > 0
    has_label = stats["has_labels"] and len(stats["label_counts"]) > 0
    n_cols = 1 + int(has_label) + int(has_ext)
    title = f"Chunk statistics{' — ' + label if label else ''}"

    fig = plt.figure(figsize=(6.5 * n_cols, 9))
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.30, wspace=0.28)
    col = 0

    # Column: chunk size distribution (top %, bottom count)
    n_chunks = max(len(pts), 1)
    bins = min(50, max(1, len(pts) // 2 + 1))
    mean_v, med_v = float(pts.mean()), float(np.median(pts))

    ax = fig.add_subplot(gs[0, col])
    ax.hist(pts, bins=bins, weights=np.full(len(pts), 100.0 / n_chunks),
            color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(mean_v, color="tomato", linestyle="--", label=f"mean = {mean_v:,.0f}")
    ax.axvline(med_v, color="goldenrod", linestyle=":", label=f"median = {med_v:,.0f}")
    ax.set_ylabel("Chunks (%)")
    ax.set_title("Chunk size distribution")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, col]); col += 1
    ax.hist(pts, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(mean_v, color="tomato", linestyle="--")
    ax.axvline(med_v, color="goldenrod", linestyle=":")
    ax.set_xlabel("Points per chunk")
    ax.set_ylabel("Number of chunks")

    # Column: label distribution, bars colored per classs
    if has_label:
        lc = stats["label_counts"]
        names = list(stats.get("label_names") or [])
        names += [f"cls {i}" for i in range(len(names), len(lc))]
        bar_colors = _colors_to_mpl(stats.get("label_colors"), len(lc))
        total_valid = max(int(lc.sum()), 1)
        xs = list(range(len(lc)))

        ax = fig.add_subplot(gs[0, col])
        ax.bar(xs, lc / total_valid * 100, color=bar_colors, edgecolor="black", linewidth=0.3)
        ax.set_ylabel("Scored points (%)")
        ax.set_title("Label distribution")
        ax.set_xticks(xs)
        ax.set_xticklabels(names[: len(lc)], rotation=45, ha="right", fontsize=7)

        ax = fig.add_subplot(gs[1, col]); col += 1
        ax.bar(xs, lc, color=bar_colors, edgecolor="black", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_ylabel("Total points (log)")
        ax.set_xticks(xs)
        ax.set_xticklabels(names[: len(lc)], rotation=45, ha="right", fontsize=7)

        ignore = stats.get("ignore_index")
        n_ignored = stats.get("n_ignored_points", 0)
        if ignore is not None and n_ignored:
            total = lc.sum() + n_ignored
            ax.set_xlabel(f"ignored points: {n_ignored:,} ({n_ignored/total*100:.1f} % of all)")

    # Column: spatial extent (no % analog → span both rows)
    if has_ext:
        ext = stats["extents"]
        ax = fig.add_subplot(gs[:, col]); col += 1
        ax.boxplot(
            [ext[:, 0], ext[:, 1], ext[:, 2]],
            labels=["X", "Y", "Z"],
            patch_artist=True,
            boxprops=dict(facecolor="#5cb85c", alpha=0.7),
            medianprops=dict(color="white", linewidth=2),
        )
        ax.set_ylabel("Extent (m)")
        ax.set_title("Spatial extent per chunk")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[chunk_stats] Figure saved → {fig_path}")

# Coverage stats — text + figure

def _format_coverage_stats(stats: dict, label: str = "") -> str:
    lines = []
    sep = "─" * 56
    title = f"Coverage report{' — ' + label if label else ''}"
    lines += [f"\n{sep}", title, sep]

    ns = stats["n_chunks"]
    nif = stats["n_chunks_image_free"]
    nnk = stats.get("n_chunks_no_image_key", 0)
    lines.append(f"  Chunks examined            : {ns:>10,}")
    lines.append(f"  Image-free chunks (0 cams) : {nif:>10,}  ({nif/ns*100 if ns else 0:5.1f} % of chunks)")
    if nnk:
        lines.append(f"  Chunks w/o image data      : {nnk:>10,}  (no image arrays at all)")

    tp = stats["total_points"]
    cp = stats["covered_points"]
    up = stats["uncovered_points"]
    cr = stats["coverage_rate"]
    lines.append(f"  Total LiDAR points         : {tp:>10,}")
    lines.append(f"  Points seen by ≥1 camera   : {cp:>10,}  ({cr*100:5.1f} %)")
    lines.append(f"  Points seen by no camera   : {up:>10,}  ({up/tp*100 if tp else 0:5.1f} %)")

    hist = stats["coverage_histogram"]
    if len(hist) > 1:
        lines.append("\n  Camera multiplicity (how many cameras see each point):")
        for k, cnt in enumerate(hist):
            bar = "█" * int(cnt / tp * 40) if tp else ""
            lines.append(f"    {k:2d} camera(s): {cnt:>10,}  ({cnt/tp*100 if tp else 0:5.1f} %)  {bar}")

    pc = stats["per_cam_covered"]
    pr = stats["per_cam_rate"]
    if len(pc):
        lines.append(f"\n  Per camera-slot coverage (share of all {tp:,} points seen by that slot):")
        for i, (cnt, rate) in enumerate(zip(pc, pr)):
            bar = "█" * int(rate * 40)
            lines.append(f"    slot {i:2d}: {cnt:>10,}  ({rate*100:5.1f} %)  {bar}")

    rates = stats["per_sample_rates"]
    valid = rates[~np.isnan(rates)]
    if len(valid):
        lines.append("\n  Per-chunk coverage rate (fraction of a chunk's points seen by ≥1 camera):")
        lines.append(
            f"    min={valid.min()*100:.1f}%  "
            f"mean={valid.mean()*100:.1f}%  "
            f"median={np.median(valid)*100:.1f}%  "
            f"max={valid.max()*100:.1f}%"
        )

    if stats["chunks_below"]:
        n_with = ns - nif - nnk
        lines.append("\n  Chunks below coverage threshold (excluding image-free chunks):")
        for t, cnt in stats["chunks_below"].items():
            lines.append(
                f"    < {t*100:4.0f}%: {cnt:>6,} / {n_with:,}"
                f"  ({cnt/n_with*100 if n_with else 0:.1f} %)"
            )
    lines.append(sep)
    return "\n".join(lines)


def print_coverage_stats(stats: dict, label: str = "") -> None:
    print(_format_coverage_stats(stats, label))


def _save_coverage_stats_figure(stats: dict, fig_path, label: str = "") -> None:
    """Two-row dashboard: percentages on top, absolute counts on the bottom.

    Columns: per-chunk coverage rate, camera multiplicity, per-camera-slot
    coverage. Image-free chunks are excluded from the per-chunk rate (reported
    separately in the text/printout).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig_path = Path(fig_path)
    rates = stats["per_sample_rates"]
    valid = rates[~np.isnan(rates)]
    pc = stats["per_cam_covered"]
    pr = stats["per_cam_rate"]
    hist = stats["coverage_histogram"]
    tp = max(stats["total_points"], 1)

    has_percam = len(pc) > 0
    has_hist = len(hist) > 1
    n_cols = 1 + int(has_hist) + int(has_percam)
    title = f"Coverage report{' — ' + label if label else ''}"

    fig = plt.figure(figsize=(6.5 * n_cols, 9))
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.30, wspace=0.28)
    col = 0

    # Column: per-chunk coverage rate (top %, bottom count)
    nb = len(valid)
    bins = min(50, max(1, nb // 5 + 1))
    ax = fig.add_subplot(gs[0, col])
    if nb:
        ax.hist(valid * 100, bins=bins, weights=np.full(nb, 100.0 / nb),
                color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(float(valid.mean() * 100), color="tomato", linestyle="--", label=f"mean = {valid.mean()*100:.1f}%")
        ax.axvline(float(np.median(valid) * 100), color="goldenrod", linestyle=":", label=f"median = {np.median(valid)*100:.1f}%")
        ax.legend(fontsize=8)
    ax.set_ylabel("Chunks (%)")
    ax.set_title("Per-chunk coverage rate\n(chunks with ≥1 camera)")

    ax = fig.add_subplot(gs[1, col]); col += 1
    if nb:
        ax.hist(valid * 100, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(float(valid.mean() * 100), color="tomato", linestyle="--")
        ax.axvline(float(np.median(valid) * 100), color="goldenrod", linestyle=":")
    ax.set_xlabel("Coverage rate (%)")
    ax.set_ylabel("Number of chunks")

    # Column: camera multiplicity (top % of points, bottom count)
    if has_hist:
        ks = np.arange(len(hist))
        ax = fig.add_subplot(gs[0, col])
        ax.bar(ks, hist / tp * 100, color="#5bc0de", edgecolor="white", linewidth=0.4)
        ax.set_ylabel("Points (%)")
        ax.set_title("Camera multiplicity")
        ax.set_xticks(ks)

        ax = fig.add_subplot(gs[1, col]); col += 1
        ax.bar(ks, hist, color="#5bc0de", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Number of cameras covering a point")
        ax.set_ylabel("Number of points")
        ax.set_xticks(ks)

    #Column: per-camera-slot coverage (top %, bottom count)
    if has_percam:
        xs = list(range(len(pc)))
        ax = fig.add_subplot(gs[0, col])
        ax.bar(xs, pr * 100, color="#5cb85c", edgecolor="white", linewidth=0.4)
        ax.set_ylabel("Points seen (%)")
        ax.set_title("Per-camera-slot coverage")
        ax.set_ylim(0, 100)
        ax.set_xticks(xs)

        ax = fig.add_subplot(gs[1, col]); col += 1
        ax.bar(xs, pc, color="#5cb85c", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Camera slot")
        ax.set_ylabel("Number of points")
        ax.set_xticks(xs)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[coverage_stats] Figure saved → {fig_path}")

# Image-pass stats (`<split>_image_stats.json`, Helimap only)

def load_image_pass_stats(data_root, split: str) -> dict | None:
    """Load `<split>_image_stats.json` written by helimap_preprocess_images.py.

    Returns None if the file doesn't exist.
    """
    path = Path(data_root) / f"{split}_image_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _format_image_pass_stats(stats: dict, label: str = "") -> str:
    lines = []
    sep = "─" * 56
    title = f"Image-pass report (preprocessing){' — ' + label if label else ''}"
    lines += [f"\n{sep}", title, sep]

    lines.append(f"  Generated            : {stats.get('generated', '?')}")
    lines.append(f"  Chunks total         : {stats['n_chunks_total']:>10,}")
    lines.append(f"  Chunks saved         : {stats['n_chunks_saved']:>10,}")
    lines.append(f"  Chunks no camera     : {stats['n_chunks_no_camera']:>10,}")
    lines.append(f"  Chunks skipped       : {stats['n_chunks_skipped']:>10,}  (already done)")

    tp = stats["total_pts"]
    cp = stats["covered_pts"]
    up = stats["uncovered_pts"]
    cr = stats["coverage_rate"]
    lines.append(f"\n  Points total         : {tp:>12,}")
    lines.append(f"  Covered (≥1 cam)     : {cp:>12,}  ({cr*100:5.1f} %)")
    lines.append(f"  Uncovered (0 cams)   : {up:>12,}  ({up/tp*100 if tp else 0:5.1f} %)")

    per_slot = stats.get("per_slot", {})
    if per_slot:
        lines.append("\n  Per camera-slot coverage:")
        name_w = max(len(n) for n in per_slot) + 2
        for name, d in per_slot.items():
            rate = d["coverage_rate"]
            bar = "█" * int(rate * 30)
            lines.append(f"    {name:<{name_w}}: {d['covered_pts']:>10,}  ({rate*100:5.1f} %)  {bar}")

    crop_summary = stats.get("crop_stats", {})
    if crop_summary:
        lines.append("\n  Crop / tight-box distribution (px):")
        for type_tag, e in crop_summary.items():
            lines.append(f"\n    [{type_tag}]  {e['n_crops']:,} crops")
            hdr = f"      {'metric':<22}  {'min':>6}  {'p10':>6}  {'med':>6}  {'p90':>6}  {'max':>6}  {'mean':>7}"
            lines.append(hdr)
            for field, fname in [("tight_w", "tight box W"), ("tight_h", "tight box H"),
                                  ("crop_w", "crop window W"), ("crop_h", "crop window H"),
                                  ("n_vis", "visible pts/crop")]:
                v = e[field]
                lines.append(f"      {fname:<22}  {v['min']:>6}  {v['p10']:>6}  {v['med']:>6}"
                              f"  {v['p90']:>6}  {v['max']:>6}  {v['mean']:>7.0f}")
            lines.append(f"      min_crop_px was binding in {e['min_crop_enforced_pct']:.1f}% of crops")

    lines.append(sep)
    return "\n".join(lines)


def print_image_pass_stats(stats: dict, label: str = "") -> None:
    print(_format_image_pass_stats(stats, label))


def _save_image_pass_figure(stats: dict, fig_path, label: str = "") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig_path = Path(fig_path)
    per_slot = stats.get("per_slot", {})
    crop_summary = stats.get("crop_stats", {})

    has_percam = len(per_slot) > 0
    has_crops = len(crop_summary) > 0
    n_cols = int(has_percam) + int(has_crops)
    if n_cols == 0:
        return

    title = f"Image-pass report (preprocessing){' — ' + label if label else ''}"
    fig = plt.figure(figsize=(7 * n_cols, 5))
    gs = gridspec.GridSpec(1, n_cols, figure=fig)
    col = 0

    if has_percam:
        ax = fig.add_subplot(gs[0, col]); col += 1
        names = list(per_slot.keys())
        rates = [d["coverage_rate"] * 100 for d in per_slot.values()]
        colors = ["#5cb85c" if n.startswith("front") else "#f0ad4e" for n in names]
        ax.bar(range(len(names)), rates, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Per-camera-slot coverage")
        ax.set_ylim(0, 100)

    if has_crops:
        ax = fig.add_subplot(gs[0, col]); col += 1
        data, labels_, colors = [], [], []
        palette = {"front": "#5cb85c", "nadir": "#f0ad4e"}
        for type_tag, e in crop_summary.items():
            for field in ("tight_w", "tight_h", "crop_w", "crop_h"):
                data.append([e[field]["min"], e[field]["p10"], e[field]["med"],
                              e[field]["p90"], e[field]["max"]])
                labels_.append(f"{type_tag}\n{field}")
                colors.append(palette.get(type_tag, "#5bc0de"))
        # boxplot from 5-number summaries: draw as bars with whiskers via errorbar
        x = np.arange(len(data))
        meds = [d[2] for d in data]
        lo = [d[2] - d[1] for d in data]
        hi = [d[3] - d[2] for d in data]
        ax.bar(x, meds, color=colors, edgecolor="white", linewidth=0.4)
        ax.errorbar(x, meds, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_, fontsize=7)
        ax.set_ylabel("pixels (median, p10–p90)")
        ax.set_title("Crop / tight-box size distribution")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[image_pass_stats] Figure saved → {fig_path}")



# Top-level driver

def analyze_split(dataset_name: str, path: str, split: str, out_dir: Path,
                  with_image: bool = True, coverage_thresholds=(0.10, 0.25, 0.50),
                  num_workers: int = 0, sweeps: int = 10):
    print(f"\n{'='*70}\nDataset: {dataset_name}   split: {split}\n{'='*70}")

    name = dataset_name.lower()
    dataset = _build_dataset(dataset_name, path, split, with_image=with_image, sweeps=sweeps)
    label_names = getattr(dataset, "CLASS_NAMES", None) or _FALLBACK_CLASS_NAMES.get(name)
    label_colors = getattr(dataset, "CLASS_COLORS", None) or _FALLBACK_CLASS_COLORS.get(name)
    ignore_index = _IGNORE_INDEX.get(name)

    chunk_stats, coverage_stats = compute_stats(
        dataset, label_names=label_names, label_colors=label_colors,
        ignore_index=ignore_index,
        coverage_thresholds=coverage_thresholds, num_workers=num_workers,
    )

    print_chunk_stats(chunk_stats, label=split)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}_chunk_stats.txt").write_text(_format_chunk_stats(chunk_stats, split))
    _save_chunk_stats_figure(chunk_stats, out_dir / f"{split}_chunk_stats.png", label=split)

    if with_image:
        print_coverage_stats(coverage_stats, label=split)
        (out_dir / f"{split}_coverage_stats.txt").write_text(_format_coverage_stats(coverage_stats, split))
        _save_coverage_stats_figure(coverage_stats, out_dir / f"{split}_coverage_stats.png", label=split)

    image_pass_stats = load_image_pass_stats(path, split)
    if image_pass_stats is not None:
        print_image_pass_stats(image_pass_stats, label=split)
        (out_dir / f"{split}_image_pass_stats.txt").write_text(_format_image_pass_stats(image_pass_stats, split))
        _save_image_pass_figure(image_pass_stats, out_dir / f"{split}_image_pass_stats.png", label=split)
    elif with_image:
        print(f"\n  (no {split}_image_stats.json found in {path} — skipping preprocessing image-pass report)")


def main():
    parser = argparse.ArgumentParser(description="Analyze preprocessed chunk datasets (GridNet / Atocha / A9CR / nuScenes)")
    parser.add_argument("--dataset", required=True, choices=["gridnet", "atocha", "a9cr", "nuscenes"])
    parser.add_argument("--path", required=True, help="Root dir containing <split>/ chunk folders")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to analyze (default: train val test)")
    parser.add_argument("--out-dir", default="analysis", help="Directory to write reports/figures into")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip loading camera images (faster; only chunk stats are computed)")
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Parallel DataLoader workers for chunk loading (0 = serial). "
                             "Set to match the CPUs reserved for the job.")
    parser.add_argument("--sweeps", type=int, default=10,
                        help="nuScenes only: number of aggregated sweeps; selects the info pkl "
                             "(nuscenes_infos_{sweeps}sweeps_{split}.pkl). Ignored for other datasets.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for split in args.splits:
        try:
            analyze_split(
                args.dataset, args.path, split, out_dir,
                with_image=not args.no_images,
                coverage_thresholds=tuple(args.coverage_thresholds),
                num_workers=args.num_workers,
                sweeps=args.sweeps,
            )
        except RuntimeError as e:
            print(f"\n[{split}] skipped: {e}")


if __name__ == "__main__":
    main()
