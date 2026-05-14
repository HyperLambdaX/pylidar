"""Synthetic forest LAS generator for demos and end-to-end tests.

Used by ``tests/test_demo_e2e.py`` to produce a reproducible small LAS at
runtime (Q5 decision, 2026-05-11). Pure numpy + laspy; no new runtime
deps. The script can also be invoked as a CLI for ad-hoc generation:

    uv run tools/gen_synthetic_las.py /tmp/forest.las \\
        --n-trees 5 --area 20 --point-format 3

Generated LAS layout:

* Trees: ``n_trees`` randomly-placed (deterministic seed) circular tree
  blobs with radius ``tree_radius``, height ``tree_height`` minus a linear
  falloff with radius plus Gaussian jitter, and a ``return_number==1``
  flag (so demos exercising ``--keep-first`` keep them).
* Ground: ``n_ground`` points uniformly scattered over the bbox at
  ``z ~ N(0, 0.05)``, classified as 2 (ground).
* RGB-capable point formats get a flat colour so the writer round-trip
  exercises the RGB code path. Formats without RGB (0, 1, 4, 6, 9)
  carry only the basic + (optionally) gps_time fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import laspy
import numpy as np


_RGB_POINT_FORMATS = frozenset({2, 3, 5, 7, 8, 10})
_GPSTIME_POINT_FORMATS = frozenset({1, 3, 4, 5, 6, 7, 8, 9, 10})


def make_forest_las(
    out_path: Path,
    *,
    n_trees: int = 5,
    n_points_per_tree: int = 200,
    n_ground: int = 200,
    area: float = 25.0,
    tree_radius: float = 3.0,
    tree_height: float = 14.0,
    point_format: int = 3,
    version: str = "1.2",
    seed: int = 0,
    crs_epsg: Optional[int] = None,
) -> Path:
    """Write a small synthetic forest LAS to ``out_path`` and return the path.

    Parameters mirror the rd.md row-13 requirement: small (≤ 1k points),
    3-5 fake trees, deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)

    # Place tree centres on a jittered grid so we always get well-separated
    # trees regardless of n_trees (Poisson-disk-ish without the import).
    rows = max(int(np.ceil(np.sqrt(n_trees))), 1)
    cols = max(int(np.ceil(n_trees / rows)), 1)
    cell = area / max(rows, cols)
    centres: list[tuple[float, float]] = []
    for t in range(n_trees):
        r = t // cols
        c = t % cols
        cx = (c + 0.5) * cell + rng.uniform(-0.2, 0.2) * cell
        cy = (r + 0.5) * cell + rng.uniform(-0.2, 0.2) * cell
        centres.append((cx, cy))

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    return_numbers: list[int] = []
    classifications: list[int] = []
    for cx, cy in centres:
        radii = rng.uniform(0.0, tree_radius, size=n_points_per_tree)
        thetas = rng.uniform(0.0, 2 * np.pi, size=n_points_per_tree)
        dx = radii * np.cos(thetas)
        dy = radii * np.sin(thetas)
        h = tree_height - (tree_height / tree_radius) * radii + rng.normal(
            0, 0.3, size=n_points_per_tree
        )
        h = np.clip(h, 0.5, tree_height + 2.0)
        xs.extend((cx + dx).tolist())
        ys.extend((cy + dy).tolist())
        zs.extend(h.tolist())
        return_numbers.extend([1] * n_points_per_tree)
        classifications.extend([5] * n_points_per_tree)  # 5 = high vegetation

    if n_ground > 0:
        gx = rng.uniform(0.0, area, size=n_ground)
        gy = rng.uniform(0.0, area, size=n_ground)
        gz = rng.normal(0.0, 0.05, size=n_ground)
        xs.extend(gx.tolist())
        ys.extend(gy.tolist())
        zs.extend(gz.tolist())
        return_numbers.extend([1] * n_ground)
        classifications.extend([2] * n_ground)  # 2 = ground

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    zs_arr = np.asarray(zs, dtype=np.float64)
    rn_arr = np.asarray(return_numbers, dtype=np.uint8)
    cls_arr = np.asarray(classifications, dtype=np.uint8)
    n = xs_arr.shape[0]

    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = (0.001, 0.001, 0.001)
    header.offsets = (0.0, 0.0, 0.0)
    if crs_epsg is not None:
        try:
            header.add_crs(f"EPSG:{int(crs_epsg)}")  # laspy 2.5+
        except AttributeError:
            # Older laspy: fall through with an empty CRS; the writer
            # test still verifies VLR count parity against the input.
            pass

    las = laspy.LasData(header)
    las.x = xs_arr
    las.y = ys_arr
    las.z = zs_arr
    las.return_number = rn_arr
    las.number_of_returns = np.ones(n, dtype=np.uint8)
    las.classification = cls_arr
    las.intensity = np.zeros(n, dtype=np.uint16)

    if point_format in _GPSTIME_POINT_FORMATS:
        # Strictly-monotonic non-zero gps_time so uniqueness="gpstime"
        # has data to work on.
        las.gps_time = (np.arange(n, dtype=np.float64) + 1.0) * 1e-3 + 1000.0
    if point_format in _RGB_POINT_FORMATS:
        las.red = np.zeros(n, dtype=np.uint16)
        las.green = np.zeros(n, dtype=np.uint16)
        las.blue = np.zeros(n, dtype=np.uint16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    return out_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output", type=Path)
    ap.add_argument("--n-trees", type=int, default=5)
    ap.add_argument("--n-points-per-tree", type=int, default=200)
    ap.add_argument("--n-ground", type=int, default=200)
    ap.add_argument("--area", type=float, default=25.0)
    ap.add_argument("--tree-radius", type=float, default=3.0)
    ap.add_argument("--tree-height", type=float, default=14.0)
    ap.add_argument("--point-format", type=int, default=3)
    ap.add_argument("--version", default="1.2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crs-epsg", type=int, default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    import sys

    args = _parse_args(sys.argv[1:])
    path = make_forest_las(
        args.output,
        n_trees=args.n_trees,
        n_points_per_tree=args.n_points_per_tree,
        n_ground=args.n_ground,
        area=args.area,
        tree_radius=args.tree_radius,
        tree_height=args.tree_height,
        point_format=args.point_format,
        version=args.version,
        seed=args.seed,
        crs_epsg=args.crs_epsg,
    )
    print(f"wrote {path}")
