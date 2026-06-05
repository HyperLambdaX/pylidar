"""lidR-aligned Dalponte 2016 individual-tree-segmentation demo.

This script mirrors the Dalponte example in lidR's
``R/algorithm-its.R::dalponte2016``:

    chm <- rasterize_canopy(las, 0.5, p2r(0.3))
    ker <- matrix(1, 3, 3)
    chm <- terra::focal(chm, w = ker, fun = mean, na.rm = TRUE)
    ttops <- locate_trees(chm, lmf(4, 2))
    las <- segment_trees(las, dalponte2016(chm, ttops))

Output is a LAS/LAZ file carrying a persistent ``treeID`` extra-byte
dimension and RGB colouring when the input point format already supports
RGB. Optional CSV/NPY exports expose the detected treetops and CHM.

Run:
    uv run examples/dalponte2016_demo.py <input.las> <output_dir>
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


_RGB_POINT_FORMATS = frozenset({2, 3, 5, 7, 8, 10})


def _valid_tree_mask(tree_id: np.ndarray) -> np.ndarray:
    if np.issubdtype(tree_id.dtype, np.integer):
        na = np.iinfo(tree_id.dtype).max
        return (tree_id > 0) & (tree_id != na)
    if np.issubdtype(tree_id.dtype, np.floating):
        na = np.finfo(tree_id.dtype).tiny
        return (tree_id > 0) & (tree_id != na) & np.isfinite(tree_id)
    return np.zeros(tree_id.shape, dtype=bool)


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
    rng = np.random.default_rng(seed)
    h = (np.arange(k) + 0.5) / max(k, 1)
    rng.shuffle(h)
    s = rng.uniform(0.55, 0.95, size=k)
    v = rng.uniform(0.75, 1.00, size=k)
    return _hsv_to_rgb(h, s, v)


def _rgb_for_tree_id(tree_id: np.ndarray, seed: int = 0) -> np.ndarray:
    rgb = np.full((tree_id.shape[0], 3), 0.4, dtype=np.float64)
    valid = _valid_tree_mask(tree_id)
    if not valid.any():
        return (rgb * 65535.0).astype(np.uint16)
    n_trees = int(tree_id[valid].max())
    palette = _random_colors(max(n_trees, 1), seed=seed)
    rgb[valid] = palette[tree_id[valid].astype(np.int64) - 1]
    return (rgb * 65535.0).clip(0, 65535).astype(np.uint16)


def _rgb_for_writer(tree_id: np.ndarray, las_in: laspy.LasData) -> np.ndarray | None:
    if int(las_in.point_format.id) in _RGB_POINT_FORMATS:
        return _rgb_for_tree_id(tree_id)
    return None


def _read_filter_kwargs(
    *,
    keep_first: bool,
    drop_class: Optional[Sequence[int]],
    drop_z_below: Optional[float],
) -> dict:
    kwargs: dict = {}
    if keep_first:
        kwargs["keep_first"] = True
    if drop_class:
        kwargs["drop_class"] = list(drop_class)
    if drop_z_below is not None:
        kwargs["drop_z_below"] = float(drop_z_below)
    return kwargs


def _layout_crs(las: laspy.LasData):
    try:
        return las.header.parse_crs()
    except Exception:
        return None


def _export_treetops_csv(path: Path, treetops: "pylidar.locate_trees.Treetops") -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "z", "tree_id"])
        for x, y, z, tid in zip(treetops.x, treetops.y, treetops.z, treetops.tree_id):
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", int(tid)])


def _export_chm(
    out_dir: Path,
    *,
    raw_chm: np.ndarray,
    smoothed_chm: np.ndarray,
    layout: "pylidar.raster.RasterLayout",
    params: dict,
) -> None:
    np.save(out_dir / "chm_raw.npy", raw_chm)
    np.save(out_dir / "chm.npy", smoothed_chm)
    sidecar = {
        "shape": [int(layout.nrow), int(layout.ncol)],
        "xmin": float(layout.xmin),
        "xmax": float(layout.xmax),
        "ymin": float(layout.ymin),
        "ymax": float(layout.ymax),
        "xres": float(layout.xres),
        "yres": float(layout.yres),
        "nrow": int(layout.nrow),
        "ncol": int(layout.ncol),
        "crs_wkt": layout.crs.to_wkt() if layout.crs is not None else None,
        "nodata": "nan",
        "params": params,
    }
    with (out_dir / "chm.json").open("w") as fh:
        json.dump(sidecar, fh, indent=2)


def _summary(dt: float, tree_id: np.ndarray, n_seeds: int) -> None:
    valid = _valid_tree_mask(tree_id)
    n_trees = int(tree_id[valid].max()) if valid.any() else 0
    frac = float(valid.mean()) if tree_id.size else 0.0
    print(
        f"  dalponte2016  {dt:.2f}s - {n_trees} trees, "
        f"{n_seeds} seeds, {100.0 * frac:.1f}% assigned"
    )


def main(
    in_path: Path,
    out_dir: Path,
    *,
    keep_first: bool = False,
    drop_class: Optional[Sequence[int]] = None,
    drop_z_below: Optional[float] = None,
    res: float = 0.5,
    subcircle: float = 0.3,
    focal_size: int = 3,
    ws: float = 4.0,
    hmin: float = 2.0,
    th_tree: float = 2.0,
    th_seed: float = 0.45,
    th_cr: float = 0.55,
    max_cr: float = 10.0,
    uniqueness: str = "incremental",
    export_treetops: bool = False,
    export_chm: bool = False,
) -> None:
    """Run the lidR-shaped Dalponte workflow on one LAS/LAZ file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    read_kwargs = _read_filter_kwargs(
        keep_first=keep_first,
        drop_class=drop_class,
        drop_z_below=drop_z_below,
    )
    label = ", ".join(f"{k}={v}" for k, v in read_kwargs.items()) or "no filters"
    print(f"reading {in_path} ({label})")
    las_in = pylidar.io.read_las(str(in_path), **read_kwargs)
    if len(las_in.x) == 0:
        raise SystemExit("filters rejected every point - nothing to segment")

    xyz = np.ascontiguousarray(np.stack([
        np.asarray(las_in.x, dtype=np.float64),
        np.asarray(las_in.y, dtype=np.float64),
        np.asarray(las_in.z, dtype=np.float64),
    ], axis=1))
    print(
        f"  {xyz.shape[0]} points; z range "
        f"{float(xyz[:, 2].min()):.2f} .. {float(xyz[:, 2].max()):.2f} m"
    )

    print(
        "\n[dalponte2016] p2r(subcircle={:.3g}) -> focal mean {}x{} -> "
        "lmf(ws={}m, hmin={}m) -> dalponte2016".format(
            subcircle, focal_size, focal_size, ws, hmin
        )
    )
    t0 = time.perf_counter()

    layout = pylidar.raster.RasterLayout.from_lidr_extent(
        xyz, res=float(res), crs=_layout_crs(las_in)
    )
    raw_chm = pylidar.raster.rasterize_canopy_p2r(
        xyz, layout, subcircle=float(subcircle)
    )
    chm = np.ascontiguousarray(
        pylidar.raster.focal_mean(raw_chm, window_size=int(focal_size)),
        dtype=np.float64,
    )

    treetops = pylidar.locate_trees.locate_trees_chm(
        chm,
        layout,
        ws=float(ws),
        hmin=float(hmin),
    )
    labels = pylidar.segmentation.dalponte2016_from_treetops(
        chm=chm,
        layout=layout,
        treetops=treetops,
        th_tree=float(th_tree),
        th_seed=float(th_seed),
        th_cr=float(th_cr),
        max_cr=float(max_cr),
    )
    tree_id = pylidar.segment_trees(xyz, layout, raster_labels=labels)

    out_path = out_dir / "dalponte2016.las"
    pylidar.io.write_las_with_treeid(
        las_in,
        tree_id,
        out_path,
        rgb=_rgb_for_writer(tree_id, las_in),
        uniqueness=uniqueness,
    )
    dt = time.perf_counter() - t0
    _summary(dt, tree_id, treetops.n)

    if export_treetops:
        _export_treetops_csv(out_dir / "treetops_dalponte2016.csv", treetops)
    if export_chm:
        _export_chm(
            out_dir,
            raw_chm=raw_chm,
            smoothed_chm=chm,
            layout=layout,
            params={
                "res": float(res),
                "subcircle": float(subcircle),
                "focal_size": int(focal_size),
                "ws": float(ws),
                "hmin": float(hmin),
                "th_tree": float(th_tree),
                "th_seed": float(th_seed),
                "th_cr": float(th_cr),
                "max_cr": float(max_cr),
            },
        )

    print(f"\ndone - wrote {out_path}")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the lidR-aligned Dalponte 2016 ITS workflow."
    )
    ap.add_argument("input", type=Path, help="input .las / .laz file")
    ap.add_argument("output_dir", type=Path, help="output directory")
    ap.add_argument("--keep-first", action="store_true")
    ap.add_argument("--drop-class", type=int, nargs="*", default=None, metavar="N")
    ap.add_argument("--drop-z-below", type=float, default=None, metavar="Z")
    ap.add_argument("--res", type=float, default=0.5, metavar="M")
    ap.add_argument("--subcircle", type=float, default=0.3, metavar="M")
    ap.add_argument("--focal-size", type=int, default=3, metavar="N")
    ap.add_argument("--ws", type=float, default=4.0, metavar="M")
    ap.add_argument("--hmin", type=float, default=2.0, metavar="M")
    ap.add_argument("--th-tree", type=float, default=2.0)
    ap.add_argument("--th-seed", type=float, default=0.45)
    ap.add_argument("--th-cr", type=float, default=0.55)
    ap.add_argument("--max-cr", type=float, default=10.0)
    ap.add_argument(
        "--uniqueness",
        choices=("incremental", "gpstime", "bitmerge"),
        default="incremental",
        help="treeID uniqueness mode used when writing LAS output",
    )
    ap.add_argument("--export-treetops", action="store_true")
    ap.add_argument("--export-chm", action="store_true")
    return ap.parse_args(list(argv))


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(
        args.input.expanduser(),
        args.output_dir.expanduser(),
        keep_first=args.keep_first,
        drop_class=args.drop_class,
        drop_z_below=args.drop_z_below,
        res=args.res,
        subcircle=args.subcircle,
        focal_size=args.focal_size,
        ws=args.ws,
        hmin=args.hmin,
        th_tree=args.th_tree,
        th_seed=args.th_seed,
        th_cr=args.th_cr,
        max_cr=args.max_cr,
        uniqueness=args.uniqueness,
        export_treetops=args.export_treetops,
        export_chm=args.export_chm,
    )
