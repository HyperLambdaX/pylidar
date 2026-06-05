"""Li et al. 2012 point-based individual-tree-segmentation demo.

Unlike the CHM-based Dalponte demo, ``li2012`` segments directly in 3D: it
region-grows trees from the highest remaining point using horizontal-distance
rules, so it does not collapse the scene to a 2.5D canopy surface or vacuum
whole vertical columns into one crown. That makes it the better fit for dense
terrestrial / mobile (MLS) scans such as power-line corridors.

Pipeline (normalize ON by default, recommended for raw absolute elevations):

    read_las
      -> optional crop
      -> classify_ground (CSF default) -> normalize_height (tin)
      -> voxel-downsample to a tractable budget (li2012 is O(N^2))
      -> li2012(xyz_norm) on the downsampled subset
      -> propagate subset treeIDs back to every original point (3D nearest)
      -> optional low-height treeID mask
      -> write_las_with_treeid(original LAS)

Run:
    uv run examples/li2012_demo.py <input.las> <output_dir>

Tip for big clouds: li2012 cost grows with the SEGMENTED point count, not the
input size, because the demo downsamples first. Tune ``--target-points`` (or set
``--voxel`` explicitly) to trade speed for detail; use ``--crop`` to focus on a
region of interest.
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
from scipy.spatial import cKDTree

import pylidar


_RGB_POINT_FORMATS = frozenset({2, 3, 5, 7, 8, 10})


# ----------------------------------------------------------------- treeID utils
def _valid_tree_mask(tree_id: np.ndarray) -> np.ndarray:
    if np.issubdtype(tree_id.dtype, np.integer):
        na = np.iinfo(tree_id.dtype).max
        return (tree_id > 0) & (tree_id != na)
    if np.issubdtype(tree_id.dtype, np.floating):
        na = np.finfo(tree_id.dtype).tiny
        return (tree_id > 0) & (tree_id != na) & np.isfinite(tree_id)
    return np.zeros(tree_id.shape, dtype=bool)


def _treeid_na_value(tree_id: np.ndarray) -> int | float:
    if np.issubdtype(tree_id.dtype, np.integer):
        return int(np.iinfo(tree_id.dtype).max)
    if np.issubdtype(tree_id.dtype, np.floating):
        return float(np.finfo(tree_id.dtype).tiny)
    raise TypeError("tree_id must be integer or floating")


def _mask_treeid_below(
    tree_id: np.ndarray, z_norm: np.ndarray, threshold: float | None
) -> np.ndarray:
    if threshold is None:
        return tree_id
    out = tree_id.copy()
    out[z_norm < float(threshold)] = _treeid_na_value(out)
    return out


# ------------------------------------------------------------------- colouring
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


def _write_rgb_las(
    las_in: laspy.LasData, tree_id: np.ndarray, out_path: Path, *, seed: int = 0
) -> None:
    """Write a viewer-friendly LAS with baked per-tree RGB.

    The faithful output keeps the input's point format (often format 0, no
    RGB), so CloudCompare can only colour it by the ``treeID`` scalar — and the
    NA sentinel (2^31-1) then crushes every real tree ID into one bin. This
    companion file upgrades to an RGB-capable format and bakes a distinct
    colour per tree (grey for masked / unassigned points) so individual crowns
    are visible immediately in RGB mode, no scalar-range fiddling required.
    """
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = las_in.header.scales
    header.offsets = las_in.header.offsets
    try:  # carry CRS when present so coordinates stay georeferenced
        crs = las_in.header.parse_crs()
        if crs is not None:
            header.add_crs(crs)
    except Exception:
        pass
    out = laspy.LasData(header)
    out.x = np.asarray(las_in.x)
    out.y = np.asarray(las_in.y)
    out.z = np.asarray(las_in.z)
    rgb = _rgb_for_tree_id(tree_id, seed=seed)
    out.red, out.green, out.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    out.write(str(out_path))


# --------------------------------------------------------------------- reading
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


def _derive_ground_candidate_mask(las: laspy.LasData) -> tuple[np.ndarray, str]:
    n = len(las.x)
    try:
        rn = np.asarray(las.return_number)
        nr = np.asarray(las.number_of_returns)
    except Exception:
        return np.ones(n, dtype=np.bool_), "all_points"
    if rn.shape[0] != n or nr.shape[0] != n:
        return np.ones(n, dtype=np.bool_), "all_points"
    mask = (nr > 0) & (rn == nr)
    if not mask.any():
        return np.ones(n, dtype=np.bool_), "all_points"
    return np.ascontiguousarray(mask, dtype=np.bool_), "last_returns"


# ------------------------------------------------------------ subset / propagate
def _crop_mask(
    xyz: np.ndarray, crop: Optional[Sequence[float]]
) -> np.ndarray:
    n = xyz.shape[0]
    if not crop:
        return np.ones(n, dtype=bool)
    xmin, ymin, xmax, ymax = (float(v) for v in crop)
    return (
        (xyz[:, 0] >= xmin)
        & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] <= ymax)
    )


def _voxel_keep_indices(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one representative point per voxel (the highest z).

    Keeping the tallest point per voxel preserves apexes, which is what the
    li2012 top-down region grower keys off.
    """
    if voxel <= 0.0:
        return np.arange(xyz.shape[0], dtype=np.int64)
    origin = xyz.min(axis=0)
    keys = np.floor((xyz - origin) / voxel).astype(np.int64)
    # Stable order: tallest point first within each voxel, then unique-by-voxel.
    order = np.argsort(-xyz[:, 2], kind="stable")
    keys_sorted = keys[order]
    _, first = np.unique(keys_sorted, axis=0, return_index=True)
    return np.sort(order[first])


def _auto_voxel_for_target(xyz: np.ndarray, target: int) -> tuple[np.ndarray, float]:
    """Binary-search a voxel size that downsamples to ~target points."""
    n = xyz.shape[0]
    if n <= target:
        return np.arange(n, dtype=np.int64), 0.0
    span = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    lo, hi = 1e-3, max(span, 1e-3)
    best_idx = _voxel_keep_indices(xyz, hi)
    best_v = hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        idx = _voxel_keep_indices(xyz, mid)
        if idx.shape[0] > target:
            lo = mid  # too many points -> bigger voxel
        else:
            best_idx, best_v = idx, mid
            hi = mid  # few enough -> try finer
    return best_idx, best_v


def _propagate_labels(
    sub_xyz: np.ndarray, sub_labels: np.ndarray, full_xyz: np.ndarray
) -> np.ndarray:
    """Assign every full-cloud point the treeID of its nearest segmented point."""
    tree = cKDTree(sub_xyz)
    _, idx = tree.query(full_xyz, k=1)
    return sub_labels[idx].astype(np.int32)


# ------------------------------------------------------------------- exporting
def _tree_apices(xyz: np.ndarray, tree_id: np.ndarray) -> list[tuple[float, float, float, int]]:
    """Per-tree highest point, as a stand-in for explicit treetops."""
    valid = _valid_tree_mask(tree_id)
    rows: list[tuple[float, float, float, int]] = []
    for tid in np.unique(tree_id[valid]):
        m = tree_id == tid
        zi = int(np.argmax(xyz[m, 2]))
        sub = xyz[m]
        rows.append((float(sub[zi, 0]), float(sub[zi, 1]), float(sub[zi, 2]), int(tid)))
    return rows


def _export_apices_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "z", "tree_id"])
        for x, y, z, tid in rows:
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", int(tid)])


# ------------------------------------------------------------------------- main
def main(
    in_path: Path,
    out_dir: Path,
    *,
    keep_first: bool = False,
    drop_class: Optional[Sequence[int]] = None,
    drop_z_below: Optional[float] = None,
    crop: Optional[Sequence[float]] = None,
    normalize: bool = True,
    ground_method: str = "csf",
    csf_class_threshold: float = 0.5,
    csf_cloth_resolution: float = 0.5,
    csf_rigidness: int = 1,
    csf_iterations: int = 500,
    csf_time_step: float = 0.65,
    normalize_method: str = "tin",
    target_points: int = 60000,
    voxel: float = 0.0,
    max_points: int = 200000,
    dt1: float = 1.5,
    dt2: float = 2.0,
    R: float = 2.0,
    Zu: float = 15.0,
    hmin: float = 2.0,
    speed_up: float = 10.0,
    mask_treeid_z_below: float | None = None,
    uniqueness: str = "incremental",
    export_treetops: bool = False,
) -> None:
    """Run the li2012 point-based ITS workflow on one LAS/LAZ file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    read_kwargs = _read_filter_kwargs(
        keep_first=keep_first, drop_class=drop_class, drop_z_below=drop_z_below
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

    crop_mask = _crop_mask(xyz, crop)
    if not crop_mask.all():
        print(f"  crop -> {int(crop_mask.sum())} points kept")
    if not crop_mask.any():
        raise SystemExit("crop window kept no points")

    t0 = time.perf_counter()

    # ---- normalize (optional) ----
    meta: dict = {
        "normalize": bool(normalize),
        "ground_algorithm": None,
        "ground_candidate_source": None,
        "ground_points": None,
        "normalize_method": None,
    }
    xyz_seg = xyz.copy()
    if normalize:
        candidate_mask, candidate_source = _derive_ground_candidate_mask(las_in)
        if ground_method == "csf":
            algo = pylidar.ground.csf(
                class_threshold=float(csf_class_threshold),
                cloth_resolution=float(csf_cloth_resolution),
                rigidness=int(csf_rigidness),
                iterations=int(csf_iterations),
                time_step=float(csf_time_step),
            )
        elif ground_method == "pmf":
            gnd_ws, gnd_th = pylidar.ground.util_makeZhangParam()
            algo = pylidar.ground.pmf(ws=gnd_ws, th=gnd_th)
        else:
            raise ValueError("ground_method must be 'csf' or 'pmf'")
        ground_mask = pylidar.ground.classify_ground(
            xyz=xyz, algorithm=algo, candidate_mask=candidate_mask
        )
        z_norm = pylidar.normalize.normalize_height(
            xyz=xyz, ground_mask=ground_mask, method=normalize_method  # type: ignore[arg-type]
        )
        xyz_seg[:, 2] = z_norm
        meta.update(
            ground_algorithm=ground_method,
            ground_candidate_source=candidate_source,
            ground_points=int(ground_mask.sum()),
            normalize_method=normalize_method,
        )
        print(
            f"  normalized with {ground_method}/{normalize_method}: "
            f"{int(ground_mask.sum())} ground points; z_norm range "
            f"{float(z_norm.min()):.2f} .. {float(z_norm.max()):.2f} m"
        )
    z_for_mask = xyz_seg[:, 2]
    effective_mask_threshold = (
        0.0 if normalize and mask_treeid_z_below is None else mask_treeid_z_below
    )

    # ---- downsample for tractable li2012 ----
    seg_pool = np.flatnonzero(crop_mask)
    pool_xyz = np.ascontiguousarray(xyz_seg[seg_pool])
    if voxel > 0.0:
        keep_local = _voxel_keep_indices(pool_xyz, float(voxel))
        used_voxel = float(voxel)
    else:
        keep_local, used_voxel = _auto_voxel_for_target(pool_xyz, int(target_points))
    sub_xyz = np.ascontiguousarray(pool_xyz[keep_local])
    n_sub = sub_xyz.shape[0]
    print(
        f"  downsample for li2012: {pool_xyz.shape[0]} -> {n_sub} points "
        f"(voxel={used_voxel:.3g} m)"
    )
    if n_sub > max_points:
        raise SystemExit(
            f"segment set has {n_sub} points (> --max-points {max_points}). "
            f"li2012 is O(N^2); increase --voxel, lower --target-points, or --crop."
        )
    if n_sub < 2:
        raise SystemExit("too few points to segment after downsampling")

    # ---- li2012 ----
    print(
        f"\n[li2012] dt1={dt1} dt2={dt2} R={R} Zu={Zu} hmin={hmin} "
        f"speed_up={speed_up} on {n_sub} points"
    )
    sub_labels = pylidar.segmentation.li2012(
        xyz=sub_xyz,
        dt1=float(dt1),
        dt2=float(dt2),
        R=float(R),
        Zu=float(Zu),
        hmin=float(hmin),
        speed_up=float(speed_up),
    )
    n_trees = int(np.unique(sub_labels[sub_labels > 0]).size)

    # ---- propagate to full cloud ----
    tree_id_full = np.zeros(xyz.shape[0], dtype=np.int32)
    tree_id_full[seg_pool] = _propagate_labels(sub_xyz, sub_labels, pool_xyz)
    tree_id = _mask_treeid_below(tree_id_full, z_for_mask, effective_mask_threshold)

    out_path = out_dir / "li2012.las"
    pylidar.io.write_las_with_treeid(
        las_in,
        tree_id,
        out_path,
        rgb=_rgb_for_writer(tree_id, las_in),
        uniqueness=uniqueness,
    )
    # Always emit a viewer-friendly RGB copy so per-tree colours show up
    # directly in CloudCompare even when the input format has no RGB channel.
    rgb_path = out_dir / "li2012_rgb.las"
    _write_rgb_las(las_in, tree_id, rgb_path)
    dt = time.perf_counter() - t0
    frac = float(_valid_tree_mask(tree_id).mean()) if tree_id.size else 0.0
    print(
        f"  li2012  {dt:.2f}s - {n_trees} trees, "
        f"{100.0 * frac:.1f}% of points assigned (after masking)"
    )

    if export_treetops:
        rows = _tree_apices(xyz, tree_id)
        _export_apices_csv(out_dir / "treetops_li2012.csv", rows)

    sidecar = {
        **meta,
        "n_trees": n_trees,
        "segment_points": int(n_sub),
        "voxel": float(used_voxel),
        "target_points": int(target_points),
        "params": {
            "dt1": float(dt1), "dt2": float(dt2), "R": float(R),
            "Zu": float(Zu), "hmin": float(hmin), "speed_up": float(speed_up),
        },
        "mask_treeid_z_below": effective_mask_threshold,
        "uniqueness": uniqueness,
    }
    with (out_dir / "li2012.json").open("w") as fh:
        json.dump(sidecar, fh, indent=2)

    print(f"\ndone - wrote {out_path}")
    print(f"       wrote {rgb_path} (per-tree RGB; open this in CloudCompare)")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the lidR-aligned Li 2012 point-based ITS workflow."
    )
    ap.add_argument("input", type=Path, help="input .las / .laz file")
    ap.add_argument("output_dir", type=Path, help="output directory")
    # read filters
    ap.add_argument("--keep-first", action="store_true")
    ap.add_argument("--drop-class", type=int, nargs="*", default=None, metavar="N")
    ap.add_argument("--drop-z-below", type=float, default=None, metavar="Z")
    ap.add_argument(
        "--crop", type=float, nargs=4, default=None,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="restrict segmentation to a bounding box",
    )
    # normalize / ground
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip ground classification and height normalization")
    ap.add_argument("--ground-method", choices=("csf", "pmf"), default="csf")
    ap.add_argument("--csf-class-threshold", type=float, default=0.5)
    ap.add_argument("--csf-cloth-resolution", type=float, default=0.5)
    ap.add_argument("--csf-rigidness", type=int, default=1)
    ap.add_argument("--csf-iterations", type=int, default=500)
    ap.add_argument("--csf-time-step", type=float, default=0.65)
    ap.add_argument("--normalize-method", choices=("tin", "knnidw", "kriging"),
                    default="tin")
    # downsample budget
    ap.add_argument("--target-points", type=int, default=60000,
                    help="auto voxel-downsample to ~this many points before li2012")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="explicit voxel size (m); overrides --target-points")
    ap.add_argument("--max-points", type=int, default=200000,
                    help="hard cap on the li2012 input (O(N^2) guard)")
    # li2012 params
    ap.add_argument("--dt1", type=float, default=1.5)
    ap.add_argument("--dt2", type=float, default=2.0)
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--Zu", type=float, default=15.0)
    ap.add_argument("--hmin", type=float, default=2.0)
    ap.add_argument("--speed-up", type=float, default=10.0)
    # output
    ap.add_argument("--mask-treeid-z-below", type=float, default=None, metavar="Z")
    ap.add_argument("--uniqueness",
                    choices=("incremental", "gpstime", "bitmerge"),
                    default="incremental")
    ap.add_argument("--export-treetops", action="store_true")
    return ap.parse_args(list(argv))


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(
        args.input.expanduser(),
        args.output_dir.expanduser(),
        keep_first=args.keep_first,
        drop_class=args.drop_class,
        drop_z_below=args.drop_z_below,
        crop=args.crop,
        normalize=not args.no_normalize,
        ground_method=args.ground_method,
        csf_class_threshold=args.csf_class_threshold,
        csf_cloth_resolution=args.csf_cloth_resolution,
        csf_rigidness=args.csf_rigidness,
        csf_iterations=args.csf_iterations,
        csf_time_step=args.csf_time_step,
        normalize_method=args.normalize_method,
        target_points=args.target_points,
        voxel=args.voxel,
        max_points=args.max_points,
        dt1=args.dt1,
        dt2=args.dt2,
        R=args.R,
        Zu=args.Zu,
        hmin=args.hmin,
        speed_up=args.speed_up,
        mask_treeid_z_below=args.mask_treeid_z_below,
        uniqueness=args.uniqueness,
        export_treetops=args.export_treetops,
    )
