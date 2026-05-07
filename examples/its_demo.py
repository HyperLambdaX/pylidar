"""Mirror of the lidR README "Individual tree segmentation" minimal demo.

lidR's demo is essentially:

    las <- readLAS("<file.las>")
    las <- segment_trees(las, li2012())
    col <- random.colors(200)
    plot(las, color = "treeID", colorPalette = col)

This script does the same with pylidar:
1. read MixedConifer.laz with laspy
2. run pylidar.segmentation.li2012 with default parameters
3. paint each point with a random color keyed by its treeID
4. write a LAS (point format 3, with RGB) so it can be opened in CloudCompare

Run:
    uv run examples/its_demo.py \\
        /Users/lambdayin/Downloads/MixedConifer.laz \\
        /Users/lambdayin/Downloads/MixedConifer_segmented.las
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import laspy
import numpy as np

from pylidar.segmentation import li2012


def random_colors(k: int, seed: int = 0) -> np.ndarray:
    """k distinct RGB colors in [0, 1], roughly mirroring lidR's random.colors.

    Pastel-ish: hue is shuffled across the wheel, saturation and value stay
    high enough to keep individual trees visually separable in CloudCompare.
    """
    rng = np.random.default_rng(seed)
    h = (np.arange(k) + 0.5) / k
    rng.shuffle(h)
    s = rng.uniform(0.55, 0.95, size=k)
    v = rng.uniform(0.75, 1.00, size=k)
    return _hsv_to_rgb(h, s, v)


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


def main(in_path: Path, out_path: Path) -> None:
    print(f"reading {in_path}")
    las_in = laspy.read(str(in_path))
    n = len(las_in.x)
    print(f"  {n} points; z range {float(las_in.z.min()):.2f} .. {float(las_in.z.max()):.2f}")

    xyz = np.ascontiguousarray(
        np.stack([np.asarray(las_in.x, dtype=np.float64),
                  np.asarray(las_in.y, dtype=np.float64),
                  np.asarray(las_in.z, dtype=np.float64)], axis=1)
    )

    print("running li2012 (defaults: dt1=1.5, dt2=2.0, R=2, Zu=15, hmin=2, speed_up=10)")
    t0 = time.perf_counter()
    tree_id = li2012(xyz=xyz)
    dt = time.perf_counter() - t0
    n_trees = int(tree_id.max())
    n_assigned = int((tree_id > 0).sum())
    print(f"  {dt:.1f}s, detected {n_trees} trees, "
          f"{n_assigned}/{n} points assigned ({100 * n_assigned / n:.1f}%)")

    # one random color per detected tree id; id 0 (= unsegmented points,
    # below hmin or otherwise rejected) gets neutral grey.
    palette = random_colors(max(n_trees, 1), seed=0)
    rgb = np.full((n, 3), 0.4, dtype=np.float64)
    mask = tree_id > 0
    rgb[mask] = palette[tree_id[mask] - 1]
    rgb16 = (rgb * 65535.0).clip(0, 65535).astype(np.uint16)

    # build a fresh LAS with point format 3 so we can store RGB. point format
    # 1 (the input) has GPS time but no color; format 3 has both.
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = las_in.header.scales
    header.offsets = las_in.header.offsets
    las_out = laspy.LasData(header)
    las_out.x = las_in.x
    las_out.y = las_in.y
    las_out.z = las_in.z
    las_out.intensity = las_in.intensity
    las_out.classification = las_in.classification
    if hasattr(las_in, "gps_time"):
        las_out.gps_time = las_in.gps_time
    las_out.red = rgb16[:, 0]
    las_out.green = rgb16[:, 1]
    las_out.blue = rgb16[:, 2]

    print(f"writing {out_path}")
    las_out.write(str(out_path))
    print("done — open in CloudCompare to inspect tree-coloured point cloud")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input.laz> <output.las>")
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
