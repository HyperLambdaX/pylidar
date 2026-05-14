"""Multi-algorithm ITS demo for pylidar.

Mirrors the lidR README's individual-tree-segmentation pattern across every
algorithm currently ported in pylidar. Four LAS files are written, one per
segmentation algorithm; each output carries a persistent ``treeID`` LAS
extra-byte dimension (Phase 5) and RGB colouring per tree so CloudCompare
can render the result without any plotting dependency.

Algorithm coverage (4 segmentation algos + 3 helpers, all exercised):

    li2012        — point-cloud-only ITS, no preprocessing
    silva2016     — point-cloud + treetops; treetops via lmf_points
    dalponte2016  — CHM + seeds; seeds via lmf_chm, CHM via p2r on
                    chm_smooth-smoothed point heights
    watershed     — CHM-only; reuses the smoothed CHM

Phase 6 CLI surface (all flags optional, sensible defaults):

    --keep-first              read only first-return points (lidR
                              ``filter = "-keep_first"``)
    --drop-class INT [...]    drop points with these classifications
    --drop-z-below FLOAT      drop points below this z value
    --ws FLOAT                local-maximum filter window (m), default 5.0
    --hmin FLOAT              minimum tree-top height (m), default 2.0
    --export-treetops         write CSV with treetop x/y/z/tree_id per algorithm
    --export-chm              write CHM npy + JSON sidecar with layout info

Run:
    uv run examples/its_demo.py <input.laz> <output_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

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


_RGB_POINT_FORMATS = frozenset({2, 3, 5, 7, 8, 10})


def _valid_tree_mask(tree_id: np.ndarray) -> np.ndarray:
    """Bool mask of points that actually belong to a tree.

    Excludes both pylidar's in-grid ``0 == no tree`` convention AND the
    LAS-spec NA sentinels (``np.iinfo(int32).max`` for int32 labels,
    ``np.finfo(float64).tiny`` for float64 labels). The NA sentinels can
    appear when ``segment_trees(layout=..., raster_labels=...)`` runs with
    its default ``nodata=None`` (which resolves to the LAS NA) and any
    input point falls outside the CHM bbox.
    """
    if np.issubdtype(tree_id.dtype, np.integer):
        na = np.iinfo(tree_id.dtype).max
        return (tree_id > 0) & (tree_id != na)
    if np.issubdtype(tree_id.dtype, np.floating):
        na = np.finfo(tree_id.dtype).tiny
        return (tree_id > 0) & (tree_id != na) & np.isfinite(tree_id)
    return np.zeros(tree_id.shape, dtype=bool)


def _rgb_for_tree_id(tree_id: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return an (N, 3) uint16 RGB array, one random colour per real tree.

    Points whose ``tree_id`` is the no-tree value (0) or the LAS NA
    sentinel are coloured neutral grey. Sentinel filtering is essential:
    a raw ``tree_id.max()`` on int32 NA sentinels (~2.1 billion) would
    try to allocate a 50+ GB palette.
    """
    n = tree_id.shape[0]
    valid = _valid_tree_mask(tree_id)
    rgb = np.full((n, 3), 0.4, dtype=np.float64)  # neutral grey
    if not valid.any():
        return (rgb * 65535.0).astype(np.uint16)
    # The integer label is a 1-based tree index; the palette size only
    # needs to span the max **valid** label, never the NA sentinel.
    n_trees = int(tree_id[valid].max())
    palette = _random_colors(max(n_trees, 1), seed=seed)
    rgb[valid] = palette[tree_id[valid].astype(np.int64) - 1]
    return (rgb * 65535.0).clip(0, 65535).astype(np.uint16)


def _rgb_for_writer(
    tree_id: np.ndarray, las_in: "laspy.LasData", seed: int = 0
) -> np.ndarray | None:
    """Return colours only if the input LAS point format supports RGB.

    Point formats 0/1/4/6/9 have no R/G/B channels; the writer would
    refuse to upgrade format in v0. In that case, fall through to a
    treeID-only output (no colouring).
    """
    if int(las_in.point_format.id) in _RGB_POINT_FORMATS:
        return _rgb_for_tree_id(tree_id, seed=seed)
    return None


# ----------------------------------------------------------------- exports
def _export_treetops_csv(
    out_path: Path,
    treetops_xyz: np.ndarray,
    tree_ids: np.ndarray,
) -> None:
    """Write (x, y, z, tree_id) rows to ``out_path``."""
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "z", "tree_id"])
        for (x, y, z), tid in zip(treetops_xyz, tree_ids):
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", int(tid)])


def _export_chm_npy(
    out_dir: Path,
    chm: np.ndarray,
    layout: "pylidar.raster.RasterLayout",
) -> None:
    """Write ``chm.npy`` and a ``chm.json`` sidecar with layout metadata.

    JSON sidecar carries enough information for an external script to
    reconstruct the cartographic transform without a GeoTIFF dep.
    """
    np.save(out_dir / "chm.npy", chm)
    sidecar = {
        "shape": [int(layout.nrow), int(layout.ncol)],
        "xmin": float(layout.xmin),
        "ymax": float(layout.ymax),
        "xres": float(layout.xres),
        "yres": float(layout.yres),
        "nrow": int(layout.nrow),
        "ncol": int(layout.ncol),
        "crs_wkt": layout.crs.to_wkt() if layout.crs is not None else None,
        "nodata": "nan",
    }
    with (out_dir / "chm.json").open("w") as fh:
        json.dump(sidecar, fh, indent=2)


# ----------------------------------------------------------------- driver
def _summary(tag: str, dt: float, tree_id: np.ndarray) -> None:
    """Print a per-algorithm result line.

    Counts and assigned-fraction are computed over the NA-aware valid
    mask (see :func:`_valid_tree_mask`) so the LAS NA sentinel doesn't
    leak into the headline numbers.
    """
    if tree_id.size == 0:
        n_trees = 0
        frac = 0.0
    else:
        valid = _valid_tree_mask(tree_id)
        n_trees = int(tree_id[valid].max()) if valid.any() else 0
        frac = float(valid.mean())
    print(f"  {tag}  {dt:.2f}s — {n_trees} trees, {100 * frac:.1f}% assigned")


def _read_filter_kwargs(
    *,
    keep_first: bool,
    drop_class: Optional[Sequence[int]],
    drop_z_below: Optional[float],
) -> dict:
    """Translate demo CLI flags into ``pylidar.io.read_las`` kwargs."""
    kwargs: dict = {}
    if keep_first:
        kwargs["keep_first"] = True
    if drop_class:
        kwargs["drop_class"] = list(drop_class)
    if drop_z_below is not None:
        kwargs["drop_z_below"] = float(drop_z_below)
    return kwargs


def main(
    in_path: Path,
    out_dir: Path,
    *,
    keep_first: bool = False,
    drop_class: Optional[Sequence[int]] = None,
    drop_z_below: Optional[float] = None,
    ws: float = 5.0,
    hmin: float = 2.0,
    export_treetops: bool = False,
    export_chm: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    read_kwargs = _read_filter_kwargs(
        keep_first=keep_first,
        drop_class=drop_class,
        drop_z_below=drop_z_below,
    )
    filter_label = ", ".join(f"{k}={v}" for k, v in read_kwargs.items()) or "no filters"
    print(f"reading {in_path}  ({filter_label})")
    las_in = pylidar.io.read_las(str(in_path), **read_kwargs)
    n = len(las_in.x)
    if n == 0:
        raise SystemExit("filters rejected every point — nothing to segment")
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
    point_labels = pylidar.segmentation.li2012(xyz=xyz)
    tree_id = pylidar.segment_trees(xyz, point_labels=point_labels)
    dt = time.perf_counter() - t0
    pylidar.io.write_las_with_treeid(
        las_in, tree_id, out_dir / "li2012.las",
        rgb=_rgb_for_writer(tree_id, las_in, seed=0),
        uniqueness="incremental",
    )
    _summary("li2012", dt, tree_id)

    # ----- 2. silva2016 (treetops via lmf_points → silva2016) -----
    print(f"\n[2/4] silva2016 — lmf_points (ws={ws}m, hmin={hmin}m) ▶ silva2016")
    t0 = time.perf_counter()
    is_top = pylidar.segmentation.lmf_points(xyz=xyz, ws=float(ws), hmin=float(hmin))
    treetops_xyz = np.ascontiguousarray(xyz[is_top], dtype=np.float64)
    point_labels = pylidar.segmentation.silva2016(xyz=xyz, treetops=treetops_xyz)
    tree_id = pylidar.segment_trees(xyz, point_labels=point_labels)
    dt = time.perf_counter() - t0
    pylidar.io.write_las_with_treeid(
        las_in, tree_id, out_dir / "silva2016.las",
        rgb=_rgb_for_writer(tree_id, las_in, seed=0),
        uniqueness="incremental",
    )
    _summary(f"silva2016 ({int(is_top.sum())} treetops)", dt, tree_id)
    if export_treetops and treetops_xyz.size:
        # silva2016 treetops live in point-cloud space; tree_ids are 1..K
        _export_treetops_csv(
            out_dir / "treetops_silva2016.csv",
            treetops_xyz,
            np.arange(1, treetops_xyz.shape[0] + 1, dtype=np.int32),
        )

    # ----- shared CHM for the two raster-based algorithms -----
    # lidR pattern: smooth_height(las, size=3) → rasterize_canopy(las, 0.5, p2r()).
    # We use the same: smooth point z first, then build a 0.5 m CHM via p2r.
    print("\n  building shared CHM (chm_smooth size=3m ▶ p2r at 0.5m) ...")
    t0 = time.perf_counter()
    z_smooth = pylidar.segmentation.chm_smooth(xyz=xyz, size=3.0, method="average")
    xyz_smooth = np.ascontiguousarray(np.stack(
        [xyz[:, 0], xyz[:, 1], z_smooth], axis=1), dtype=np.float64)
    layout = pylidar.raster.RasterLayout.from_extent(xyz_smooth, res=0.5)
    chm = pylidar.raster.rasterize_canopy_p2r(xyz_smooth, layout)
    # dalponte / watershed cannot consume NaN; replace with 0 (below th_tree).
    chm_for_seg = np.where(np.isfinite(chm), chm, 0.0)
    print(f"  {time.perf_counter() - t0:.2f}s — chm {chm.shape}, "
          f"max {float(np.nanmax(chm)):.2f} m")
    if export_chm:
        _export_chm_npy(out_dir, chm, layout)

    # ----- 3. dalponte2016 (lmf_chm → seeds → dalponte2016 on CHM) -----
    print(f"\n[3/4] dalponte2016 — lmf_chm (ws={ws}m, hmin={hmin}m) ▶ dalponte2016")
    t0 = time.perf_counter()
    treetops = pylidar.locate_trees.locate_trees_chm(
        chm=chm_for_seg, layout=layout, ws=float(ws), hmin=float(hmin),
    )
    labels = pylidar.segmentation.dalponte2016_from_treetops(
        chm=chm_for_seg, layout=layout, treetops=treetops,
    )
    tree_id = pylidar.segment_trees(xyz, layout, raster_labels=labels)
    dt = time.perf_counter() - t0
    pylidar.io.write_las_with_treeid(
        las_in, tree_id, out_dir / "dalponte2016.las",
        rgb=_rgb_for_writer(tree_id, las_in, seed=0),
        uniqueness="incremental",
    )
    _summary(f"dalponte2016 ({treetops.tree_id.size} seeds)", dt, tree_id)
    if export_treetops and treetops.tree_id.size:
        _export_treetops_csv(
            out_dir / "treetops_dalponte2016.csv",
            np.stack([treetops.x, treetops.y, treetops.z], axis=1),
            treetops.tree_id,
        )

    # ----- 4. watershed (CHM only) -----
    print("\n[4/4] watershed — watershed on the same CHM")
    t0 = time.perf_counter()
    labels = pylidar.segmentation.watershed(chm=chm_for_seg, th_tree=2.0, tol=1.0)
    tree_id = pylidar.segment_trees(xyz, layout, raster_labels=labels)
    dt = time.perf_counter() - t0
    pylidar.io.write_las_with_treeid(
        las_in, tree_id, out_dir / "watershed.las",
        rgb=_rgb_for_writer(tree_id, las_in, seed=0),
        uniqueness="incremental",
    )
    _summary("watershed", dt, tree_id)

    print(f"\ndone — open the four .las files in {out_dir} with CloudCompare")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run every pylidar ITS algorithm on a single LAS file."
    )
    ap.add_argument("input", type=Path, help="input .las / .laz file")
    ap.add_argument("output_dir", type=Path, help="output directory (created if missing)")
    ap.add_argument(
        "--keep-first",
        action="store_true",
        help="keep only first-return points (lidR -keep_first)",
    )
    ap.add_argument(
        "--drop-class",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="drop points with these classification codes (e.g. --drop-class 2 7)",
    )
    ap.add_argument(
        "--drop-z-below",
        type=float,
        default=None,
        metavar="Z",
        help="drop points whose z is below this value (lidR -drop_z_below)",
    )
    ap.add_argument(
        "--ws",
        type=float,
        default=5.0,
        metavar="M",
        help="local-maximum filter window in metres (default 5.0)",
    )
    ap.add_argument(
        "--hmin",
        type=float,
        default=2.0,
        metavar="M",
        help="minimum tree-top height in metres (default 2.0)",
    )
    ap.add_argument(
        "--export-treetops",
        action="store_true",
        help="write treetops_<algo>.csv alongside each LAS output",
    )
    ap.add_argument(
        "--export-chm",
        action="store_true",
        help="write the shared CHM to chm.npy + chm.json sidecar",
    )
    return ap.parse_args(list(argv))


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(
        args.input.expanduser(),
        args.output_dir.expanduser(),
        keep_first=args.keep_first,
        drop_class=args.drop_class,
        drop_z_below=args.drop_z_below,
        ws=args.ws,
        hmin=args.hmin,
        export_treetops=args.export_treetops,
        export_chm=args.export_chm,
    )
