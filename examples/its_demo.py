"""Multi-algorithm ITS demo for pylidar.

Mirrors the lidR README's individual-tree-segmentation pattern across every
algorithm currently ported in pylidar. Four LAS files are written, one per
segmentation algorithm; each point is RGB-coloured by its assigned treeID
so CloudCompare can render the result without any plotting dependency.

Algorithm coverage (4 segmentation algos + 3 helpers, all exercised):

    li2012        — point-cloud-only ITS, no preprocessing
    silva2016     — point-cloud + treetops; treetops via lmf_points
    dalponte2016  — CHM + seeds; seeds via lmf_chm, CHM built on
                    chm_smooth-smoothed point heights
    watershed     — CHM-only; reuses the smoothed CHM

Run:
    uv run examples/its_demo.py <input.laz> <output_dir>

Example:
    uv run examples/its_demo.py \\
        ~/Downloads/MixedConifer.laz ~/Downloads/MixedConifer_segmented
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import laspy
import numpy as np

import pylidar


# ----------------------------------------------------------------- palette
def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def _random_colors(k: int, seed: int = 0) -> np.ndarray:
    """k pastel-ish RGB colours in [0, 1]; mirrors lidR's random.colors."""
    rng = np.random.default_rng(seed)
    h = (np.arange(k) + 0.5) / max(k, 1)
    rng.shuffle(h)
    s = rng.uniform(0.55, 0.95, size=k)
    v = rng.uniform(0.75, 1.00, size=k)
    return _hsv_to_rgb(h, s, v)


# ----------------------------------------------------------------- raster
def _build_chm(xyz: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple p2r CHM: per pixel, max z of points falling in it.

    Returns (chm, row_idx_per_point, col_idx_per_point). Empty cells are 0
    (below any reasonable th_tree), which keeps the CHM strictly float64
    with no NaN — matching what dalponte2016 / watershed expect.
    """
    xmin = float(xyz[:, 0].min())
    ymax = float(xyz[:, 1].max())
    xmax = float(xyz[:, 0].max())
    ymin = float(xyz[:, 1].min())
    w = max(int(np.ceil((xmax - xmin) / res)) + 1, 1)
    h = max(int(np.ceil((ymax - ymin) / res)) + 1, 1)
    col = np.clip(((xyz[:, 0] - xmin) / res).astype(np.int64), 0, w - 1)
    row = np.clip(((ymax - xyz[:, 1]) / res).astype(np.int64), 0, h - 1)

    flat_idx = row * w + col
    chm_flat = np.full(h * w, -np.inf, dtype=np.float64)
    np.maximum.at(chm_flat, flat_idx, xyz[:, 2])
    chm = chm_flat.reshape(h, w)
    chm = np.where(np.isfinite(chm), chm, 0.0)
    return np.ascontiguousarray(chm, dtype=np.float64), row, col


# ------------------------------------------------------------------ writer
def _write_coloured_las(
    template: "laspy.LasData",
    out_path: Path,
    tree_id: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[int, float]:
    """Write a copy of `template` painted with one random colour per treeID.

    Returns (n_trees, fraction_assigned).
    """
    n = len(template.x)
    n_trees = int(tree_id.max()) if tree_id.size else 0
    palette = _random_colors(max(n_trees, 1), seed=seed)
    rgb = np.full((n, 3), 0.4, dtype=np.float64)  # neutral grey for treeID == 0
    mask = tree_id > 0
    if mask.any():
        rgb[mask] = palette[tree_id[mask] - 1]
    rgb16 = (rgb * 65535.0).clip(0, 65535).astype(np.uint16)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = template.header.scales
    header.offsets = template.header.offsets
    out = laspy.LasData(header)
    out.x = template.x
    out.y = template.y
    out.z = template.z
    out.intensity = template.intensity
    out.classification = template.classification
    if hasattr(template, "gps_time"):
        out.gps_time = template.gps_time
    out.red = rgb16[:, 0]
    out.green = rgb16[:, 1]
    out.blue = rgb16[:, 2]
    out.write(str(out_path))
    return n_trees, mask.mean()


# ----------------------------------------------------------------- driver
def main(in_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {in_path}")
    las_in = laspy.read(str(in_path))
    n = len(las_in.x)
    xyz = np.ascontiguousarray(np.stack([
        np.asarray(las_in.x, dtype=np.float64),
        np.asarray(las_in.y, dtype=np.float64),
        np.asarray(las_in.z, dtype=np.float64),
    ], axis=1))
    print(f"  {n} points; z range {float(xyz[:, 2].min()):.2f} .. "
          f"{float(xyz[:, 2].max()):.2f} m")

    # ----- 1. li2012 (point cloud only) -----
    print("\n[1/4] li2012 — point-cloud ITS, no helpers")
    t0 = time.perf_counter()
    tree_id = pylidar.segmentation.li2012(xyz=xyz)
    dt = time.perf_counter() - t0
    n_tr, frac = _write_coloured_las(las_in, out_dir / "li2012.las", tree_id, seed=0)
    print(f"  {dt:.2f}s — {n_tr} trees, {100 * frac:.1f}% assigned")

    # ----- 2. silva2016 (treetops via lmf_points → silva2016) -----
    print("\n[2/4] silva2016 — lmf_points (ws=5m) ▶ silva2016")
    t0 = time.perf_counter()
    is_top = pylidar.segmentation.lmf_points(xyz=xyz, ws=5.0, hmin=2.0)
    treetops = np.ascontiguousarray(xyz[is_top], dtype=np.float64)
    tree_id = pylidar.segmentation.silva2016(xyz=xyz, treetops=treetops)
    dt = time.perf_counter() - t0
    n_tr, frac = _write_coloured_las(las_in, out_dir / "silva2016.las", tree_id, seed=0)
    print(f"  {dt:.2f}s — {int(is_top.sum())} treetops, {n_tr} trees retained, "
          f"{100 * frac:.1f}% assigned")

    # ----- shared CHM for the two raster-based algorithms -----
    # lidR pattern: smooth_height(las, size=3) → rasterize_canopy(las, 0.5, p2r()).
    # We do the same: smooth point z first, then build a 0.5 m CHM by max-z.
    print("\n  building shared CHM (chm_smooth size=3m ▶ p2r at 0.5m) ...")
    t0 = time.perf_counter()
    z_smooth = pylidar.segmentation.chm_smooth(xyz=xyz, size=3.0, method="average")
    xyz_smooth = np.ascontiguousarray(np.stack(
        [xyz[:, 0], xyz[:, 1], z_smooth], axis=1), dtype=np.float64)
    chm, row_idx, col_idx = _build_chm(xyz_smooth, res=0.5)
    print(f"  {time.perf_counter() - t0:.2f}s — chm {chm.shape}, "
          f"max {float(chm.max()):.2f} m")

    # ----- 3. dalponte2016 (lmf_chm → seeds → dalponte2016 on CHM) -----
    print("\n[3/4] dalponte2016 — lmf_chm (ws=10px≈5m) ▶ dalponte2016")
    t0 = time.perf_counter()
    peaks = pylidar.segmentation.lmf_chm(chm=chm, ws=10.0, hmin=2.0)
    seeds = np.zeros_like(chm, dtype=np.int32)
    seeds[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1, dtype=np.int32)
    labels = pylidar.segmentation.dalponte2016(chm=chm, seeds=seeds)
    tree_id = labels[row_idx, col_idx]
    dt = time.perf_counter() - t0
    n_tr, frac = _write_coloured_las(las_in, out_dir / "dalponte2016.las", tree_id, seed=0)
    print(f"  {dt:.2f}s — {len(peaks)} seeds, {n_tr} trees, "
          f"{100 * frac:.1f}% assigned")

    # ----- 4. watershed (CHM only) -----
    print("\n[4/4] watershed — watershed on the same CHM")
    t0 = time.perf_counter()
    labels = pylidar.segmentation.watershed(chm=chm, th_tree=2.0, tol=1.0)
    tree_id = labels[row_idx, col_idx]
    dt = time.perf_counter() - t0
    n_tr, frac = _write_coloured_las(las_in, out_dir / "watershed.las", tree_id, seed=0)
    print(f"  {dt:.2f}s — {n_tr} trees, {100 * frac:.1f}% assigned")

    print(f"\ndone — open the four .las files in {out_dir} with CloudCompare")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input.laz> <output_dir>")
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
