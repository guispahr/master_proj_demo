"""
reconstruct_predictions.py

Reconstruct per-chunk model predictions back to the original LAS point order
for an entire zone (or all zones of a split), then save as NPZ files ready
for leaderboard submission.

Predictions are saved as ``{area}.npz`` (key ``data``, dtype ``uint8``) — the
format expected by the GridNet leaderboard.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_CLASSES = 12   # training IDs 0..11


# Core reconstruction

def reconstruct_zone(
    chunks_dir: Path,
    pred_dir: Path,
    out_path: Path,
    ignore_index: int = 11,
) -> np.ndarray:
    """Reconstruct predictions for one zone and save as NPZ.

    Parameters
    ----------
    chunks_dir    : preprocessed zone directory (contains chunk_XXXXX/ subdirs
                    each with orig_idx.npy and meta.json)
    pred_dir      : tester output directory (contains chunk_XXXXX.npy files)
    out_path      : output path; saved as .npz (key 'data', uint8) regardless
                    of extension — use e.g. ``t1z4.npz``
    ignore_index  : label assigned to original LAS points not covered by any
                    chunk (should not occur with min_pts=1)

    Returns
    -------
    pred_orig : (N_orig,) uint8 — prediction for every original LAS point,
                in original LAS row order
    """
    chunks_dir = Path(chunks_dir)
    pred_dir   = Path(pred_dir)
    out_path   = Path(out_path)

    chunk_dirs = sorted(chunks_dir.glob("chunk_*"))
    if not chunk_dirs:
        raise RuntimeError(f"No chunk_* directories found in {chunks_dir}")

    n_las_pts = None
    for cd in chunk_dirs:
        meta_path = cd / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_las_pts = meta.get("n_las_pts")
            if n_las_pts is not None:
                break

    if n_las_pts is None:
        raise RuntimeError(
            "Could not read n_las_pts from any chunk meta.json. "
            "Re-run preprocessing with the updated gridnet_preprocess.py."
        )

    # vote_counts[i, c] = number of chunks that predicted class c for point i
    # int16 is safe: max ~32k overlapping chunks per point, far more than needed
    vote_counts = np.zeros((n_las_pts, N_CLASSES), dtype=np.int16)
    n_loaded = 0
    zone = chunks_dir.name   # e.g. "t1z6a"

    for chunk_dir in chunk_dirs:
        pred_path = pred_dir / zone / f"{chunk_dir.name}.npy"
        idx_path  = chunk_dir / "orig_idx.npy"

        if not pred_path.exists():
            print(f"  [skip] no prediction file for {chunk_dir.name}")
            continue
        if not idx_path.exists():
            raise FileNotFoundError(
                f"orig_idx.npy missing in {chunk_dir}. "
                "Re-run preprocessing with the updated gridnet_preprocess.py."
            )

        orig_idx = np.load(idx_path)           # (N_chunk,) int64
        preds    = np.load(pred_path)          # (N_chunk,) int16

        if len(orig_idx) != len(preds):
            raise ValueError(
                f"{chunk_dir.name}: orig_idx has {len(orig_idx)} entries but "
                f"prediction has {len(preds)}"
            )

        # orig_idx values are unique within a single chunk, so plain indexing
        # correctly increments each (point, class) count once.
        vote_counts[orig_idx, preds.astype(np.intp)] += 1
        n_loaded += 1

    n_covered = int((vote_counts.sum(axis=1) > 0).sum())
    n_uncovered = n_las_pts - n_covered
    print(
        f"Loaded {n_loaded}/{len(chunk_dirs)} chunks → "
        f"{n_covered:,} / {n_las_pts:,} original points covered"
    )
    if n_uncovered:
        print(
            f"  WARNING: {n_uncovered:,} points not covered by any chunk "
            f"→ assigned ignore_index={ignore_index}"
        )

    pred_orig = vote_counts.argmax(axis=1).astype(np.int16)
    pred_orig[vote_counts.sum(axis=1) == 0] = ignore_index

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=pred_orig.astype(np.uint8))
    print(f"Predictions saved → {out_path}")

    return pred_orig.astype(np.uint8)

# Full-split helper (uses preprocessed directory structure)

def reconstruct_split(
    preprocessed_root: Path,
    pred_dir: Path,
    out_dir: Path,
    split: str = "test",
    ignore_index: int = 11,
):
    """Reconstruct all zones of a split.

    Parameters
    ----------
    preprocessed_root : root of the preprocessed dataset (contains manifest.json)
    pred_dir          : directory with all chunk_XXXXX.npy prediction files
    out_dir           : where to write {zone}.npz files
    """
    preprocessed_root = Path(preprocessed_root)
    manifest_path = preprocessed_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {preprocessed_root}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    zones = sorted({entry["zone"] for entry in manifest.get(split, [])})
    if not zones:
        raise RuntimeError(f"No zones found for split '{split}' in manifest.json")

    print(f"Reconstructing {len(zones)} zone(s) for split '{split}'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for zone in zones:
        chunks_dir = preprocessed_root / split / zone
        out_path   = out_dir / f"{zone}.npz"
        print(f"\n── Zone: {zone} ──")
        try:
            reconstruct_zone(
                chunks_dir=chunks_dir,
                pred_dir=pred_dir,
                out_path=out_path,
                ignore_index=ignore_index,
            )
        except Exception as exc:
            print(f"[ERROR] zone {zone}: {exc}")


# Manifest-based helper (uses chunk_manifest.json saved by the tester)

def reconstruct_from_manifest(
    manifest_path: Path,
    pred_dir: Path,
    out_dir: Path,
    ignore_index: int = 11,
):
    """Reconstruct all areas using the chunk_manifest.json saved by the tester.

    ``chunk_manifest.json`` maps ``{chunk_folder_name: absolute_chunk_dir_path}``.
    Area names are derived from the parent directory of each chunk dir.

    Parameters
    ----------
    manifest_path : path to chunk_manifest.json (typically inside pred_dir)
    pred_dir      : directory with per-chunk .npy prediction files
    out_dir       : where to write {area}.npz files
    """
    manifest_path = Path(manifest_path)
    pred_dir      = Path(pred_dir)
    out_dir       = Path(out_dir)

    with open(manifest_path) as f:
        manifest = json.load(f)   # {chunk_name: chunk_dir_path}

    # Group chunks by area (parent directory of each chunk path).
    # Derive area and chunk_name from the path value so the manifest key format
    # doesn't matter (keys are now "{area}/{chunk_name}" to avoid collisions
    # when the same chunk directory name appears in multiple areas).
    by_area: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for _key, chunk_dir_str in manifest.items():
        chunk_dir = Path(chunk_dir_str)
        area       = chunk_dir.parent.name
        chunk_name = chunk_dir.name
        by_area[area].append((chunk_name, chunk_dir))

    print(f"Found {len(by_area)} area(s): {sorted(by_area)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for area, chunks in sorted(by_area.items()):
        print(f"\n── Area: {area} ({len(chunks)} chunks) ──")

        # Read n_las_pts from the first available chunk meta.json
        n_las_pts = None
        for _, chunk_dir in chunks:
            meta_path = chunk_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                n_las_pts = meta.get("n_las_pts")
                if n_las_pts is not None:
                    break

        if n_las_pts is None:
            print(f"[ERROR] {area}: could not read n_las_pts — skipping")
            continue

        vote_counts = np.zeros((n_las_pts, N_CLASSES), dtype=np.int16)
        n_loaded = 0

        for chunk_name, chunk_dir in chunks:
            pred_path = pred_dir / area / f"{chunk_name}.npy"
            idx_path  = chunk_dir / "orig_idx.npy"

            if not pred_path.exists():
                print(f"  [skip] no prediction file for {area}/{chunk_name}")
                continue
            if not idx_path.exists():
                print(f"  [ERROR] orig_idx.npy missing in {chunk_dir} — skipping chunk")
                continue

            orig_idx = np.load(idx_path)
            preds    = np.load(pred_path)

            if len(orig_idx) != len(preds):
                print(
                    f"  [ERROR] {chunk_name}: length mismatch "
                    f"(orig_idx={len(orig_idx)}, preds={len(preds)}) — skipping"
                )
                continue

            vote_counts[orig_idx, preds.astype(np.intp)] += 1
            n_loaded += 1

        n_covered = int((vote_counts.sum(axis=1) > 0).sum())
        n_uncovered = n_las_pts - n_covered
        print(
            f"Loaded {n_loaded}/{len(chunks)} chunks → "
            f"{n_covered:,} / {n_las_pts:,} original points covered"
        )
        if n_uncovered:
            print(
                f"  WARNING: {n_uncovered:,} points not covered by any chunk "
                f"→ assigned ignore_index={ignore_index}"
            )

        pred_orig = vote_counts.argmax(axis=1).astype(np.int16)
        pred_orig[vote_counts.sum(axis=1) == 0] = ignore_index

        out_path = out_dir / f"{area}.npz"
        np.savez_compressed(out_path, data=pred_orig.astype(np.uint8))
        print(f"Predictions saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct per-chunk predictions to original LAS point order"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_zone = subparsers.add_parser("zone", help="Reconstruct a single zone → NPZ")
    p_zone.add_argument("--chunks-dir",   required=True,
                        help="Preprocessed zone directory (contains chunk_* subdirs)")
    p_zone.add_argument("--pred-dir",     required=True,
                        help="Tester output directory (contains chunk_XXXXX.npy files)")
    p_zone.add_argument("--out-path",     required=True, help="Output .npz file path")
    p_zone.add_argument("--ignore-index", type=int, default=11)

    p_split = subparsers.add_parser("split", help="Reconstruct all zones of a split → NPZ")
    p_split.add_argument("--preprocessed-root", required=True)
    p_split.add_argument("--pred-dir",           required=True)
    p_split.add_argument("--out-dir",            required=True)
    p_split.add_argument("--split",              default="test")
    p_split.add_argument("--ignore-index",       type=int, default=11)

    p_manifest = subparsers.add_parser(
        "manifest",
        help="Reconstruct all areas using chunk_manifest.json saved by the tester → NPZ"
    )
    p_manifest.add_argument("--manifest",      required=True,
                             help="Path to chunk_manifest.json")
    p_manifest.add_argument("--pred-dir",      required=True,
                             help="Tester output directory")
    p_manifest.add_argument("--out-dir",       required=True)
    p_manifest.add_argument("--ignore-index",  type=int, default=11)

    args = parser.parse_args()

    if args.command == "zone":
        reconstruct_zone(
            chunks_dir=args.chunks_dir,
            pred_dir=args.pred_dir,
            out_path=args.out_path,
            ignore_index=args.ignore_index,
        )
    elif args.command == "split":
        reconstruct_split(
            preprocessed_root=args.preprocessed_root,
            pred_dir=args.pred_dir,
            out_dir=args.out_dir,
            split=args.split,
            ignore_index=args.ignore_index,
        )
    else:
        reconstruct_from_manifest(
            manifest_path=args.manifest,
            pred_dir=args.pred_dir,
            out_dir=args.out_dir,
            ignore_index=args.ignore_index,
        )
