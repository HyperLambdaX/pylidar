"""RasterLayout — coord convention 1:1 with lidR utils_raster.R:55-77.

Verifies upper-left origin at (xmin, ymax), col=floor((x-xmin)/xres),
row=floor((ymax-y)/yres), right/bottom edge clamping, and cell-center
roundtrip.
"""

from __future__ import annotations

import numpy as np
import pyproj
import pytest

from pylidar.raster import RasterLayout


# ---------------------------------------------------------------- construction

def test_from_extent_basic():
    xyz = np.array([
        [0.0, 0.0, 1.0],
        [9.5, 4.5, 5.0],
    ], dtype=np.float64)
    layout = RasterLayout.from_extent(xyz, res=1.0)
    assert layout.xmin == 0.0
    assert layout.ymax == 4.5
    assert layout.xres == 1.0
    assert layout.yres == 1.0
    assert layout.ncol == 10  # ceil(9.5 / 1.0) = 10
    assert layout.nrow == 5   # ceil(4.5 / 1.0) = 5
    assert layout.crs is None


def test_from_extent_with_crs():
    xyz = np.array([[0.0, 0.0, 1.0], [10.0, 10.0, 2.0]], dtype=np.float64)
    crs = pyproj.CRS.from_epsg(4326)
    layout = RasterLayout.from_extent(xyz, res=1.0, crs=crs)
    assert layout.crs == crs
    assert layout.crs.to_epsg() == 4326


def test_from_extent_empty_raises():
    xyz = np.zeros((0, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        RasterLayout.from_extent(xyz, res=1.0)


def test_xmax_ymin_derived():
    layout = RasterLayout(xmin=0.0, ymax=10.0, xres=2.0, yres=2.0, ncol=5, nrow=3)
    assert layout.xmax == 10.0  # 0 + 5*2
    assert layout.ymin == 4.0   # 10 - 3*2
    assert layout.shape == (3, 5)


# ---------------------------------------------------------------- coord mapping

def test_cell_xy_to_rowcol_basic():
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    x = np.array([0.0, 0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    y = np.array([3.5, 2.5, 1.5, 0.5, 0.0], dtype=np.float64)
    row, col = layout.cell_xy_to_rowcol(x, y)
    np.testing.assert_array_equal(col, [0, 0, 1, 2, 3])
    # row=floor((4-y)/1.0): y=3.5 → 0; y=2.5 → 1; y=1.5 → 2; y=0.5 → 3; y=0 → 4 (clamped to 3)
    np.testing.assert_array_equal(row, [0, 1, 2, 3, 3])


def test_cell_xy_right_edge_clamp():
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    x = np.array([4.0], dtype=np.float64)  # x == xmax
    y = np.array([2.0], dtype=np.float64)
    row, col = layout.cell_xy_to_rowcol(x, y)
    assert col[0] == 3  # ncol-1
    assert row[0] == 2


def test_cell_xy_bottom_edge_clamp():
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    x = np.array([2.0], dtype=np.float64)
    y = np.array([0.0], dtype=np.float64)  # y == ymin
    row, col = layout.cell_xy_to_rowcol(x, y)
    assert row[0] == 3  # nrow-1
    assert col[0] == 2


def test_rowcol_to_cell_xy_centers():
    layout = RasterLayout(xmin=0.0, ymax=4.0, xres=1.0, yres=1.0, ncol=4, nrow=4)
    rows = np.array([0, 1, 2, 3], dtype=np.int64)
    cols = np.array([0, 1, 2, 3], dtype=np.int64)
    x, y = layout.rowcol_to_cell_xy(rows, cols)
    # Cell center: x = 0 + (col + 0.5)*1.0; y = 4 - (row + 0.5)*1.0
    np.testing.assert_array_equal(x, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(y, [3.5, 2.5, 1.5, 0.5])


def test_roundtrip_cell_centers():
    layout = RasterLayout(xmin=10.0, ymax=20.0, xres=0.5, yres=0.5, ncol=8, nrow=6)
    rows = np.array([0, 2, 5], dtype=np.int64)
    cols = np.array([0, 4, 7], dtype=np.int64)
    x, y = layout.rowcol_to_cell_xy(rows, cols)
    rows2, cols2 = layout.cell_xy_to_rowcol(x, y)
    np.testing.assert_array_equal(rows2, rows)
    np.testing.assert_array_equal(cols2, cols)
