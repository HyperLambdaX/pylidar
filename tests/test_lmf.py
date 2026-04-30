"""Phase 2 acceptance tests — locate_trees_lmf_chm / locate_trees_lmf_points.

Five cases per task_plan.md:
  1. 5x5 CHM with three peaks → exactly three tree tops at the right
     world XY (transform = (origin_x, origin_y, pixel_size); world coords
     hand-computed).
  2. hmin filter on the same CHM → drops the lowest peak.
  3. Three separated point-cloud clusters → three tree tops (one per
     cluster's apex).
  4. ws=0 raises ValueError (Python wrapper).
  5. CHM that is all-NaN → returns an empty (0, 3) result.

Plus dtype/shape/transform smoke tests exercising the validators.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pylidar


# Background = 1.0, peaks at (row, col) = (1,1)→8.0, (3,3)→9.0, (4,1)→7.0.
# Spacing keeps every peak isolated under a 3x3 (ws=3, pixel_size=1) window:
# the closest pair (1,1)–(3,3) is at distance sqrt(8) ≈ 2.83 in world units,
# well outside the half_ws=1.5 search radius of either peak.
def _three_peak_chm() -> np.ndarray:
    chm = np.full((5, 5), 1.0, dtype=np.float64)
    chm[1, 1] = 8.0
    chm[3, 3] = 9.0
    chm[4, 1] = 7.0
    return chm


def _row_col_to_world(transform, r, c):
    ox, oy, ps = transform
    return ox + c * ps, oy - r * ps


def test_lmf_chm_three_peaks_transform_world_xy():
    chm = _three_peak_chm()
    transform = (100.0, 200.0, 1.0)  # origin = pixel (0,0) centre

    tops = pylidar.locate_trees_lmf_chm(
        chm, transform=transform, ws=3.0, hmin=2.0, shape="circular"
    )
    assert tops.shape == (3, 3)
    assert tops.dtype == np.float64

    # Hand-compute expected world (x, y, z) for each peak. Background = 1.0
    # < hmin = 2.0, so the three peaks are the only candidates.
    expected = {
        _row_col_to_world(transform, 1, 1) + (8.0,): None,
        _row_col_to_world(transform, 3, 3) + (9.0,): None,
        _row_col_to_world(transform, 4, 1) + (7.0,): None,
    }
    got = {(float(x), float(y), float(z)) for x, y, z in tops}
    assert got == set(expected.keys())


def test_lmf_chm_hmin_filters_low_peak():
    """hmin=7.5 → only the peaks at z=8 and z=9 survive; z=7 peak is dropped."""
    chm = _three_peak_chm()
    transform = (0.0, 0.0, 1.0)

    tops = pylidar.locate_trees_lmf_chm(
        chm, transform=transform, ws=3.0, hmin=7.5, shape="circular"
    )
    assert tops.shape == (2, 3)
    z_vals = sorted(float(z) for _, _, z in tops)
    assert z_vals == [8.0, 9.0]


def test_lmf_points_three_clusters():
    """Three well-separated XY clusters, each with one obvious apex.
    LMF must return exactly three tree tops at the apex (x, y, z)."""
    rng = np.random.default_rng(seed=11)

    def cluster(cx, cy, apex_z, base_z, k=12, jitter=0.4):
        xs = rng.uniform(-jitter, jitter, k) + cx
        ys = rng.uniform(-jitter, jitter, k) + cy
        zs = np.full(k, base_z, dtype=np.float64)
        zs[0] = apex_z
        xs[0], ys[0] = cx, cy  # nail the apex onto cluster centre
        return np.stack([xs, ys, zs], axis=1)

    a = cluster(0.0,  0.0, apex_z=10.0, base_z=5.0)
    b = cluster(10.0, 0.0, apex_z=12.0, base_z=4.0)
    c = cluster(5.0,  5.0, apex_z=8.0,  base_z=3.0)
    xyz = np.ascontiguousarray(np.vstack([a, b, c]), dtype=np.float64)

    # ws=2.0 → half_ws=1.0; clusters at >5m apart → no cross-talk.
    tops = pylidar.locate_trees_lmf_points(
        xyz, ws=2.0, hmin=2.0, shape="circular"
    )
    assert tops.shape == (3, 3)
    apex_set = {(0.0, 0.0, 10.0), (10.0, 0.0, 12.0), (5.0, 5.0, 8.0)}
    got = {(float(x), float(y), float(z)) for x, y, z in tops}
    assert got == apex_set


def test_lmf_points_ws_zero_raises_value_error():
    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="ws"):
        pylidar.locate_trees_lmf_points(xyz, ws=0.0, hmin=0.0)


def test_lmf_chm_all_nan_returns_empty():
    chm = np.full((5, 5), np.nan, dtype=np.float64)
    tops = pylidar.locate_trees_lmf_chm(
        chm, transform=(0.0, 0.0, 1.0), ws=3.0, hmin=0.0, shape="circular"
    )
    assert tops.shape == (0, 3)
    assert tops.dtype == np.float64


def test_lmf_chm_square_shape_smoke():
    """Square shape exercises the bbox-from-bounding-circle fallback in
    lmf_filter_impl. Same three-peak CHM, hmin=2 → 3 peaks expected."""
    chm = _three_peak_chm()
    tops = pylidar.locate_trees_lmf_chm(
        chm, transform=(0.0, 0.0, 1.0), ws=3.0, hmin=2.0, shape="square"
    )
    assert tops.shape == (3, 3)


def test_lmf_chm_rejects_wrong_dtype():
    chm = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        pylidar.locate_trees_lmf_chm(
            chm, transform=(0.0, 0.0, 1.0), ws=3.0, hmin=2.0
        )


def test_lmf_chm_rejects_bad_transform():
    chm = np.zeros((5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="pixel_size"):
        pylidar.locate_trees_lmf_chm(
            chm, transform=(0.0, 0.0, 0.0), ws=1.0, hmin=0.0
        )
    with pytest.raises(ValueError, match="pixel_size"):
        pylidar.locate_trees_lmf_chm(
            chm, transform=(0.0, 0.0, -1.0), ws=1.0, hmin=0.0
        )
    with pytest.raises(ValueError, match=r"3 elements"):
        pylidar.locate_trees_lmf_chm(
            chm, transform=(0.0, 0.0), ws=1.0, hmin=0.0
        )


def test_lmf_chm_rejects_non_2d():
    chm = np.zeros((5,), dtype=np.float64)
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        pylidar.locate_trees_lmf_chm(
            chm, transform=(0.0, 0.0, 1.0), ws=1.0, hmin=0.0
        )


@pytest.mark.parametrize("bad_ws", [float("nan"), float("inf"), float("-inf")])
def test_lmf_points_rejects_nonfinite_ws(bad_ws):
    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="ws"):
        pylidar.locate_trees_lmf_points(xyz, ws=bad_ws, hmin=0.0)


def test_lmf_core_rejects_nan_ws_directly():
    """C++ core invariant check fires even when callers bypass Python."""
    from pylidar import _core

    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="ws"):
        _core.lmf_points(xyz, float("nan"), 0.0, 2)


@pytest.mark.parametrize("bad_hmin", [float("nan"), float("inf"), float("-inf")])
def test_lmf_core_rejects_nonfinite_hmin_directly(bad_hmin):
    """Spec §6.4: C++ core must validate hmin too. Without this, hmin=NaN
    silently makes every `zi >= hmin` comparison false → empty result with
    no error, which is exactly the silent-bypass class Phase 1 fixed for
    size/sigma."""
    from pylidar import _core

    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="hmin"):
        _core.lmf_points(xyz, 1.0, bad_hmin, 2)


def test_lmf_points_tied_heights_are_deterministic():
    """Four equal-height points within one window: the lowest-index one
    must win, every call. Regression test for the OpenMP race in the old
    `if (z == zmax && filter[pt.id])` tie-break — that version returned
    one of {1, 2, 3, 4} tops depending on which thread happened to win
    the race."""
    # Unit square at (0,0) (1,0) (0,1) (1,1), all z = 10. ws=2.5 (radius
    # ≈ 1.25) covers axial neighbours but not the diagonal — every point's
    # window contains the lower-index points adjacent to it.
    xyz = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
            [1.0, 1.0, 10.0],
        ],
        dtype=np.float64,
    )

    results = []
    for _ in range(5):
        tops = pylidar.locate_trees_lmf_points(
            xyz, ws=2.5, hmin=0.0, shape="circular"
        )
        results.append({(float(x), float(y), float(z)) for x, y, z in tops})

    # Every call returns exactly the same set, and that set is the single
    # lowest-index point (j < i tie-break rule).
    assert all(r == results[0] for r in results), \
        f"non-deterministic across calls: {results}"
    assert results[0] == {(0.0, 0.0, 10.0)}, \
        f"tie-break should pick lowest-index point, got {results[0]}"
