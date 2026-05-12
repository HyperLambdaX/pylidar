"""Treetops dataclass + locate_trees_chm/_points + uniqueness ID modes.

Covers lidR ``R/locate_trees.R:29-102`` (locate_trees dispatch + post-LMF
treetop materialization) and ``src/RcppFunction.cpp:413-440`` (bitmerge
C kernel). Inline-data tests (Phase 3 fixture-strategy convention).
"""

from __future__ import annotations

import numpy as np
import pyproj
import pytest

from pylidar.locate_trees import (
    Treetops,
    bitmerge,
    locate_trees_chm,
    locate_trees_points,
)
from pylidar.raster import RasterLayout


# ─────────────────────────────────────────────── Treetops dataclass — gates

def _ok_arrays(n=2, dtype=np.int32):
    x = np.zeros((n,), dtype=np.float64)
    y = np.zeros((n,), dtype=np.float64)
    z = np.zeros((n,), dtype=np.float64)
    tid = np.arange(1, n + 1, dtype=dtype)
    return x, y, z, tid


def test_treetops_construct_int32_incremental():
    x, y, z, tid = _ok_arrays(3, dtype=np.int32)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    assert tt.n == 3
    assert len(tt) == 3
    assert tt.tree_id.dtype == np.int32
    assert tt.crs is None


def test_treetops_construct_float64_gpstime_ids():
    x, y, z, _ = _ok_arrays(2)
    tid = np.array([300123.456, 300123.789], dtype=np.float64)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    assert tt.tree_id.dtype == np.float64


def test_treetops_rejects_wrong_xy_dtype():
    x, y, z, tid = _ok_arrays(2)
    with pytest.raises(TypeError, match="must be a 1-D float64"):
        Treetops(x=x.astype(np.float32), y=y, z=z, tree_id=tid)


def test_treetops_rejects_wrong_tree_id_dtype():
    x, y, z, _ = _ok_arrays(2)
    bad = np.array([1, 2], dtype=np.int64)
    with pytest.raises(TypeError, match="int32 .* float64"):
        Treetops(x=x, y=y, z=z, tree_id=bad)


def test_treetops_rejects_length_mismatch():
    x = np.zeros((3,), dtype=np.float64)
    y = np.zeros((3,), dtype=np.float64)
    z = np.zeros((2,), dtype=np.float64)
    tid = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="must share length"):
        Treetops(x=x, y=y, z=z, tree_id=tid)


def test_treetops_rejects_non_pyproj_crs():
    x, y, z, tid = _ok_arrays(2)
    with pytest.raises(TypeError, match="pyproj.CRS"):
        Treetops(x=x, y=y, z=z, tree_id=tid, crs="EPSG:4326")


def test_treetops_accepts_pyproj_crs():
    x, y, z, tid = _ok_arrays(2)
    crs = pyproj.CRS.from_epsg(4326)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid, crs=crs)
    assert tt.crs == crs


def test_treetops_empty_zero_trees():
    e = np.empty((0,), dtype=np.float64)
    eid = np.empty((0,), dtype=np.int32)
    tt = Treetops(x=e, y=e, z=e, tree_id=eid)
    assert tt.n == 0


# ────────────────────────────────────────────────────────── bitmerge kernel

def test_bitmerge_zero_pair_packs_to_zero():
    x = np.array([0], dtype=np.int32)
    y = np.array([0], dtype=np.int32)
    out = bitmerge(x, y)
    # uint64(0) numerically cast to double = 0.0
    assert out.dtype == np.float64
    assert out[0] == 0.0


def test_bitmerge_packs_x_high_y_low_numeric_cast():
    """1:1 with lidR src/RcppFunction.cpp:434-435 — uint64 packing then
    `static_cast<double>(z)` (numeric, not bit reinterpret)."""
    x = np.array([1], dtype=np.int32)
    y = np.array([2], dtype=np.int32)
    out = bitmerge(x, y)
    # packed uint64 = (1 << 32) | 2 = 4294967298; static_cast<double> →
    # 4294967298.0 (exactly representable, < 2^53). The audit caught a
    # prior `.view(float64)` bit-reinterpret bug that returned 2.12e-314.
    assert out[0] == 4294967298.0


def test_bitmerge_negative_inputs_use_twos_complement_then_numeric_cast():
    """memcpy of int32 → uint32 preserves bit pattern (-1 → 0xFFFFFFFF),
    then static_cast<double>(uint64) is value-preserving (with rounding
    above 2^53)."""
    x = np.array([-1, 100], dtype=np.int32)
    y = np.array([-1, 200], dtype=np.int32)
    out = bitmerge(x, y)
    # (-1, -1) → uint64 0xFFFFFFFFFFFFFFFF = 2^64 - 1; nearest double is 2^64.
    assert out[0] == float(2 ** 64)
    # (100, 200) → (100 << 32) | 200 = 429496729800 (< 2^53, exact double).
    assert out[1] == float((100 << 32) | 200)


def test_bitmerge_uniqueness_within_grid():
    """Small-coordinate ints stay below 2^53 → uint64→double cast is
    value-preserving, so distinct (x, y) packings stay distinct."""
    x = np.arange(50, dtype=np.int32)
    y = (np.arange(50, dtype=np.int32) * 7) % 11
    out = bitmerge(x, y)
    # All values < 2^53, so float comparison is safe.
    assert len(set(out.tolist())) == 50


def test_bitmerge_lidr_doc_example_numeric():
    """lidR doc (section-uniqueness.R:25-39) cites apex (10.32, 25.64)
    with scale=0.01, offset=0 → xs=1032, ys=2564. The doc claims the
    output is the bit-reinterpret (3.34e-316), but the actual C code at
    RcppFunction.cpp:435 is `static_cast<double>(z)` — numeric. We match
    the running code, not the stale doc."""
    x = np.array([1032], dtype=np.int32)
    y = np.array([2564], dtype=np.int32)
    out = bitmerge(x, y)
    expected = float((1032 << 32) | 2564)  # = 4_432_406_252_036.0
    assert out[0] == expected


def test_bitmerge_rejects_dtype_mismatch():
    x = np.array([1], dtype=np.int64)
    y = np.array([2], dtype=np.int32)
    with pytest.raises(TypeError, match="int32"):
        bitmerge(x, y)


def test_bitmerge_rejects_shape_mismatch():
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([1], dtype=np.int32)
    with pytest.raises(ValueError, match="same shape|must share"):
        bitmerge(x, y)


# ─────────────────────────────────────────────────── locate_trees_chm

def _three_peak_chm():
    """5x5 CHM with three distinct local maxima."""
    chm = np.array([
        [0, 0, 0, 0, 0],
        [0, 8, 0, 7, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float64)
    layout = RasterLayout(
        xmin=0.0, ymax=5.0, xres=1.0, yres=1.0, ncol=5, nrow=5, crs=None,
    )
    return chm, layout


def test_locate_trees_chm_finds_three_peaks_incremental():
    chm, layout = _three_peak_chm()
    tt = locate_trees_chm(chm, layout, ws=1.0, hmin=2.0)
    assert tt.n == 3
    assert tt.tree_id.dtype == np.int32
    # Sequential ids 1..3
    assert np.array_equal(np.sort(tt.tree_id), np.array([1, 2, 3], dtype=np.int32))
    # Z values match the peaks 7, 8, 9 (some order)
    assert sorted(tt.z.tolist()) == [7.0, 8.0, 9.0]


def test_locate_trees_chm_inherits_crs():
    chm, _ = _three_peak_chm()
    crs = pyproj.CRS.from_epsg(32618)
    layout = RasterLayout(
        xmin=0.0, ymax=5.0, xres=1.0, yres=1.0, ncol=5, nrow=5, crs=crs,
    )
    tt = locate_trees_chm(chm, layout, ws=1.0, hmin=2.0)
    assert tt.crs == crs


def test_locate_trees_chm_gpstime_raises_not_implemented():
    chm, layout = _three_peak_chm()
    with pytest.raises(NotImplementedError, match="CHM has no GPS"):
        locate_trees_chm(chm, layout, ws=1.0, hmin=2.0, uniqueness="gpstime")


def test_locate_trees_chm_bitmerge_requires_offset_scale():
    chm, layout = _three_peak_chm()
    with pytest.raises(ValueError, match="las_offset"):
        locate_trees_chm(chm, layout, ws=1.0, hmin=2.0, uniqueness="bitmerge")


def test_locate_trees_chm_bitmerge_with_metadata():
    chm, layout = _three_peak_chm()
    tt = locate_trees_chm(
        chm, layout, ws=1.0, hmin=2.0, uniqueness="bitmerge",
        las_offset=(0.0, 0.0), las_scale=(0.01, 0.01),
    )
    assert tt.tree_id.dtype == np.float64
    # All three IDs distinct: cell-center xs/ys at scale 0.01 stay < 2^53,
    # so the static_cast<double>(uint64) stays value-preserving and float
    # comparison is safe.
    assert len(set(tt.tree_id.tolist())) == 3


def test_locate_trees_chm_hmin_filters():
    chm, layout = _three_peak_chm()
    # hmin=8.5 keeps only the 9-peak.
    tt = locate_trees_chm(chm, layout, ws=1.0, hmin=8.5)
    assert tt.n == 1
    assert tt.z[0] == 9.0


def test_locate_trees_chm_invalid_shape_raises():
    chm, layout = _three_peak_chm()
    with pytest.raises(ValueError, match="shape"):
        locate_trees_chm(chm, layout, ws=1.0, hmin=2.0, shape="oval")


# ────────────────────────── audit-fix regressions (2026-05-12 review)

def test_locate_trees_chm_ws_in_world_units_not_pixels():
    """Audit fix: ws is now in world units, divided by xres before passing
    to _core.lmf_chm. With xres=0.5, ws=2.0 world units = 4 pixel units.
    A 1-cell-radius peak survives a ws=4-pixel window → still detected.
    A ws=10-pixel window would suppress it (would be ws=5 world units).
    """
    # 9x9 CHM, peak at center, single cell at z=8, surrounded by 0s.
    chm = np.zeros((9, 9), dtype=np.float64)
    chm[4, 4] = 8.0
    # xres = 0.5, so 9 cells × 0.5 = 4.5 m extent on each axis.
    layout = RasterLayout(
        xmin=0.0, ymax=4.5, xres=0.5, yres=0.5, ncol=9, nrow=9, crs=None,
    )
    # ws=2.0 world units = 4 pixels — peak survives, exactly 1 found.
    tt_small = locate_trees_chm(chm, layout, ws=2.0, hmin=2.0)
    assert tt_small.n == 1
    assert tt_small.z[0] == 8.0
    # ws=8.0 world units = 16 pixels — covers the whole CHM, still 1.
    tt_huge = locate_trees_chm(chm, layout, ws=8.0, hmin=2.0)
    assert tt_huge.n == 1
    # If ws were treated as pixels (the old buggy behavior), ws=2.0 would
    # be a 2-pixel window, which on a 0.5m CHM would over-detect close
    # peaks. The fact that the unit conversion is wired correctly is what
    # this test pins.


def test_locate_trees_chm_world_unit_x_y_recovered():
    """Cell-center (x, y) returned by locate_trees_chm respects xres/yres."""
    chm = np.zeros((4, 4), dtype=np.float64)
    chm[1, 2] = 5.0
    layout = RasterLayout(
        xmin=10.0, ymax=22.0, xres=0.5, yres=0.5, ncol=4, nrow=4, crs=None,
    )
    tt = locate_trees_chm(chm, layout, ws=1.0, hmin=2.0)
    assert tt.n == 1
    # cell (row=1, col=2) center: x = 10 + 2.5*0.5 = 11.25, y = 22 - 1.5*0.5 = 21.25
    assert tt.x[0] == 11.25
    assert tt.y[0] == 21.25


def test_locate_trees_chm_rejects_chm_layout_shape_mismatch():
    """Audit fix: locate_trees_chm now validates chm.shape == layout.shape."""
    chm = np.zeros((5, 5), dtype=np.float64)
    bad_layout = RasterLayout(
        xmin=0.0, ymax=10.0, xres=1.0, yres=1.0, ncol=10, nrow=10, crs=None,
    )
    with pytest.raises(ValueError, match="match layout.shape"):
        locate_trees_chm(chm, bad_layout, ws=1.0, hmin=2.0)


def test_locate_trees_chm_rejects_non_square_pixels():
    """Audit fix: locate_trees_chm raises NotImplementedError if xres != yres,
    because _core.lmf_chm assumes square pixels."""
    chm = np.zeros((5, 5), dtype=np.float64)
    layout = RasterLayout(
        xmin=0.0, ymax=10.0, xres=1.0, yres=2.0, ncol=5, nrow=5, crs=None,
    )
    with pytest.raises(NotImplementedError, match="square pixels"):
        locate_trees_chm(chm, layout, ws=1.0, hmin=2.0)


def test_locate_trees_chm_rejects_non_positive_ws():
    chm, layout = _three_peak_chm()
    with pytest.raises(ValueError, match="ws must be > 0"):
        locate_trees_chm(chm, layout, ws=0.0, hmin=2.0)


def test_locate_trees_points_gpstime_all_zero_raises():
    """Audit fix: lidR locate_trees.R:44-45 rejects all-zero gpstime."""
    xyz = _three_tree_points()
    n = xyz.shape[0]
    gps_zero = np.zeros((n,), dtype=np.float64)
    with pytest.raises(ValueError, match="not populated"):
        locate_trees_points(
            xyz, ws=2.0, hmin=2.0, uniqueness="gpstime", gpstime=gps_zero,
        )


def test_locate_trees_points_gpstime_partial_zero_ok():
    """Only an *all-zero* gpstime is rejected; a single non-zero entry is
    enough to satisfy lidR's check (`fast_countequal == npoints` would
    return false)."""
    xyz = _three_tree_points()
    n = xyz.shape[0]
    gps = np.zeros((n,), dtype=np.float64)
    gps[0] = 1.0  # one non-zero
    tt = locate_trees_points(
        xyz, ws=2.0, hmin=2.0, uniqueness="gpstime", gpstime=gps,
    )
    assert tt.tree_id.dtype == np.float64


# ──────────────────────────────────────────────── locate_trees_points

def _three_tree_points():
    """N=24 points clustered around three treetops."""
    pts = []
    centers = [(2.0, 2.0, 8.0), (7.0, 2.0, 7.0), (4.5, 7.0, 9.0)]
    for cx, cy, cz in centers:
        # peak point + 7 lower neighbors within radius
        pts.append([cx, cy, cz])
        for dx, dy in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5),
                       (-0.3, 0.3), (0.3, -0.3), (0.4, 0.4)]:
            pts.append([cx + dx, cy + dy, cz - 1.5])
    return np.asarray(pts, dtype=np.float64)


def test_locate_trees_points_finds_three_peaks():
    xyz = _three_tree_points()
    tt = locate_trees_points(xyz, ws=2.0, hmin=2.0)
    assert tt.n == 3
    # All three peak heights present
    assert sorted(tt.z.tolist()) == [7.0, 8.0, 9.0]
    assert tt.tree_id.dtype == np.int32


def test_locate_trees_points_gpstime_assigns_apex_time():
    xyz = _three_tree_points()
    n = xyz.shape[0]
    # Make gpstime monotone — apex is the first point of each cluster (i=0,8,16).
    gps = np.arange(n, dtype=np.float64) * 0.1 + 1000.0
    tt = locate_trees_points(
        xyz, ws=2.0, hmin=2.0, uniqueness="gpstime", gpstime=gps,
    )
    assert tt.tree_id.dtype == np.float64
    # gpstime 选定 apex 是 lmf_points 决定的（z=peak 的那个点），所以 ID 必是
    # gps[apex_index]。Apex 是每簇的第一个点 (i=0,8,16)。
    expected = sorted([gps[0], gps[8], gps[16]])
    assert sorted(tt.tree_id.tolist()) == expected


def test_locate_trees_points_gpstime_requires_gpstime_array():
    xyz = _three_tree_points()
    with pytest.raises(ValueError, match="gpstime"):
        locate_trees_points(xyz, ws=2.0, hmin=2.0, uniqueness="gpstime")


def test_locate_trees_points_bitmerge_with_metadata():
    xyz = _three_tree_points()
    tt = locate_trees_points(
        xyz, ws=2.0, hmin=2.0, uniqueness="bitmerge",
        las_offset=(0.0, 0.0), las_scale=(0.01, 0.01),
    )
    assert tt.tree_id.dtype == np.float64
    # Distinct treetops at small ints → values stay below 2^53 → safe to
    # compare floats directly.
    assert len(set(tt.tree_id.tolist())) == 3


def test_locate_trees_points_carries_crs():
    xyz = _three_tree_points()
    crs = pyproj.CRS.from_epsg(32618)
    tt = locate_trees_points(xyz, ws=2.0, hmin=2.0, crs=crs)
    assert tt.crs == crs


def test_locate_trees_points_empty_when_hmin_excludes_all():
    xyz = _three_tree_points()
    tt = locate_trees_points(xyz, ws=2.0, hmin=100.0)
    assert tt.n == 0
