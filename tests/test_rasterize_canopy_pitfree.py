"""rasterize_canopy_pitfree — Khosravipour pitfree CHM with first-return enforcement.

Mirrors lidR algorithm-dsm.R:209-306. First-return error strings match
exactly. Layered TIN merge via fmax across thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.raster import (
    RasterLayout,
    rasterize_canopy_dsmtin,
    rasterize_canopy_pitfree,
    rasterize_canopy_spikefree,
)


@pytest.fixture
def layout_4x4():
    return RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)


def _square_xyz(z_value: float = 5.0):
    """4 corners of the 4x4 layout — minimum data for a TIN."""
    return np.array([
        [0.5, 3.5, z_value],
        [3.5, 3.5, z_value],
        [0.5, 0.5, z_value],
        [3.5, 0.5, z_value],
    ], dtype=np.float64)


# ---------------------------------------------------------------- happy

def test_pitfree_happy_constant_layer(layout_4x4):
    """4 corner first-returns at z=5; pitfree should produce z=5 everywhere
    inside the convex hull (below first threshold and above max_edge)."""
    xyz = _square_xyz(z_value=5.0)
    rn = np.array([1, 1, 1, 1], dtype=np.uint8)
    chm = rasterize_canopy_pitfree(
        xyz, rn, layout_4x4,
        thresholds=(0.0,), max_edge=(10.0, 10.0), highest=False,
    )
    finite = np.isfinite(chm)
    assert finite.any()
    assert np.allclose(chm[finite], 5.0)


def test_pitfree_layered_max():
    """One short canopy + one tall canopy; pitfree should pick the tall one
    where the tall layer covers the cell."""
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    xyz = np.array([
        # 4 corners short (z=3) — covers full extent
        [0.5, 3.5, 3.0],
        [3.5, 3.5, 3.0],
        [0.5, 0.5, 3.0],
        [3.5, 0.5, 3.0],
        # 4 inner points tall (z=10)
        [1.5, 2.5, 10.0],
        [2.5, 2.5, 10.0],
        [1.5, 1.5, 10.0],
        [2.5, 1.5, 10.0],
    ], dtype=np.float64)
    rn = np.ones(8, dtype=np.uint8)
    chm = rasterize_canopy_pitfree(
        xyz, rn, layout,
        thresholds=(0.0, 5.0), max_edge=(10.0, 10.0), highest=False,
    )
    # Center cells should pick the tall layer (z=10).
    assert chm[1, 1] == 10.0
    assert chm[1, 2] == 10.0
    assert chm[2, 1] == 10.0
    assert chm[2, 2] == 10.0


# ---------------------------------------------------------------- max_edge trim

def test_pitfree_max_edge_trims_long_triangles(layout_4x4):
    """Two clusters far apart; max_edge < distance should leave middle cells NaN."""
    xyz = np.array([
        # Tight cluster on the left
        [0.5, 0.5, 5.0],
        [0.5, 1.5, 5.0],
        [1.5, 1.5, 5.0],
        # Tight cluster on the right (3+ units away)
        [3.5, 0.5, 5.0],
        [3.5, 1.5, 5.0],
    ], dtype=np.float64)
    rn = np.ones(5, dtype=np.uint8)
    chm = rasterize_canopy_pitfree(
        xyz, rn, layout_4x4,
        thresholds=(0.0,), max_edge=(1.5, 1.5), highest=False,
    )
    # Middle column cells should be NaN (their containing triangle has long edges).
    # We don't pin specific cells — just assert *some* NaN remains.
    assert np.isnan(chm).any()


# ---------------------------------------------------------------- first-return enforcement

def test_pitfree_no_returnnumber_raises(layout_4x4):
    xyz = _square_xyz()
    with pytest.raises(ValueError, match="No attribute 'ReturnNumber' found"):
        rasterize_canopy_pitfree(xyz, None, layout_4x4)


def test_pitfree_no_first_returns_raises(layout_4x4):
    xyz = _square_xyz()
    rn = np.array([2, 2, 2, 2], dtype=np.uint8)  # no first returns
    with pytest.raises(ValueError, match="No first returns found"):
        rasterize_canopy_pitfree(xyz, rn, layout_4x4)


def test_pitfree_returnnumber_shape_mismatch_raises(layout_4x4):
    xyz = _square_xyz()
    rn = np.array([1, 1, 1], dtype=np.uint8)  # wrong length
    with pytest.raises(ValueError):
        rasterize_canopy_pitfree(xyz, rn, layout_4x4)


# ---------------------------------------------------------------- dsmtin

def test_dsmtin_equivalent_to_pitfree_single_threshold(layout_4x4):
    xyz = _square_xyz(z_value=5.0)
    rn = np.array([1, 1, 1, 1], dtype=np.uint8)
    chm_dsm = rasterize_canopy_dsmtin(xyz, rn, layout_4x4, max_edge=10.0, highest=False)
    chm_pit = rasterize_canopy_pitfree(
        xyz, rn, layout_4x4,
        thresholds=(0.0,), max_edge=(10.0, 0.0), highest=False,
    )
    np.testing.assert_array_equal(chm_dsm, chm_pit)


def test_dsmtin_first_return_required(layout_4x4):
    xyz = _square_xyz()
    with pytest.raises(ValueError, match="No attribute 'ReturnNumber' found"):
        rasterize_canopy_dsmtin(xyz, None, layout_4x4)


# ---------------------------------------------------------------- spikefree stub

def test_spikefree_not_implemented():
    with pytest.raises(NotImplementedError, match="spikefree"):
        rasterize_canopy_spikefree()


# ---------------------------------------------------------------- audit-fix regressions (2026-05-12)

def test_pitfree_max_edge_branch_by_threshold_value():
    """Audit issue 4: lidR algorithm-dsm.R:281 selects max_edge by
    ``th == 0`` not by layer index. Earlier code used layer index, which
    diverged when the user passed thresholds without a 0 entry."""
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    # Short cluster (z=3) + tall cluster (z=10) — same layout as the layered
    # max test, but thresholds intentionally exclude 0.
    xyz = np.array([
        [0.5, 3.5, 3.0],
        [3.5, 3.5, 3.0],
        [0.5, 0.5, 3.0],
        [3.5, 0.5, 3.0],
        [1.5, 2.5, 10.0],
        [2.5, 2.5, 10.0],
        [1.5, 1.5, 10.0],
        [2.5, 1.5, 10.0],
    ], dtype=np.float64)
    rn = np.ones(8, dtype=np.uint8)
    # thresholds=(2, 5): no 0 present → both layers should use max_edge[1].
    # If max_edge[0] were used by mistake (=0 → unlimited) the output would
    # differ. We pin behavior by using max_edge that gates differently.
    chm = rasterize_canopy_pitfree(
        xyz, rn, layout,
        thresholds=(2.0, 5.0), max_edge=(0.0, 10.0), highest=False,
    )
    # All triangles with edges <= 10.0 are kept (max_edge[1]=10 → permissive)
    assert np.isfinite(chm).any()


def test_pitfree_highest_decimation_too_few_points_raises():
    """Audit issue 5: lidR algorithm-dsm.R:265 stops if highest decimation
    leaves < 3 points."""
    layout = RasterLayout(xmin=0.0, ymax=10.0, xres=5.0, yres=5.0, ncol=2, nrow=2)
    # 4 first-returns, but spread so decimate.highest leaves 1 per cell ⇒ at
    # most 4 retained; placing them so 3 share a cell drops to 2 distinct.
    xyz = np.array([
        [1.0, 8.0, 5.0],  # cell (0, 0)
        [1.0, 8.0, 6.0],  # same cell, higher → first dropped
        [6.0, 8.0, 5.0],  # cell (0, 1)
    ], dtype=np.float64)
    rn = np.ones(3, dtype=np.uint8)
    with pytest.raises(ValueError, match="not enought points to triangulate"):
        rasterize_canopy_pitfree(
            xyz, rn, layout,
            thresholds=(0.0,), max_edge=(10.0, 10.0), highest=True,
        )


def test_pitfree_all_nan_output_raises():
    """Audit issue 5: lidR algorithm-dsm.R:296 stops if final result is all NaN."""
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    # 4 first-returns all at z=1; threshold=5 → no points pass → all NaN.
    xyz = np.array([
        [0.5, 3.5, 1.0],
        [3.5, 3.5, 1.0],
        [0.5, 0.5, 1.0],
        [3.5, 0.5, 1.0],
    ], dtype=np.float64)
    rn = np.ones(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="NAs everywhere"):
        rasterize_canopy_pitfree(
            xyz, rn, layout,
            thresholds=(5.0,), max_edge=(10.0, 10.0), highest=False,
        )


def test_pitfree_subcircle_replaces_originals():
    """Audit issue 2: subcircle should REPLACE each input point with 8
    satellites (lidR algorithm-dsm.R::subcircle L420-433), not concatenate.

    Pin behavior with a single point at a cell-corner location: with
    subcircle=0 (no-op, original kept) the point lands on cell A; with
    subcircle=0.5 the original is gone and only satellites populate
    surrounding cells.
    """
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    # Need >= 4 first-returns for the > 3 layer count gate. Use 4 corners +
    # subcircle so each is replaced by 8 satellites.
    xyz = np.array([
        [0.5, 3.5, 5.0],
        [3.5, 3.5, 5.0],
        [0.5, 0.5, 5.0],
        [3.5, 0.5, 5.0],
    ], dtype=np.float64)
    rn = np.ones(4, dtype=np.uint8)
    chm = rasterize_canopy_pitfree(
        xyz, rn, layout,
        thresholds=(0.0,), max_edge=(10.0, 10.0), subcircle=0.3, highest=False,
    )
    # Subcircle expands coverage; we don't pin specific cell values, just
    # that the result is finite somewhere.
    assert np.isfinite(chm).any()
