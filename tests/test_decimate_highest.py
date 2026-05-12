"""decimate.highest — per-cell argmax with strict-< first-encountered tie-break.

Mirrors lidR src/LAS.cpp:605-617. Returns boolean mask (Python convention)
rather than 1-based index array (R convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.decimate import decimate_highest
from pylidar.raster import RasterLayout


@pytest.fixture
def layout_2x2():
    return RasterLayout(xmin=0.0, ymax=2.0, xres=1.0, yres=1.0, ncol=2, nrow=2)


def test_happy_one_per_cell(layout_2x2):
    # 6 points across 4 cells; per cell the highest survives.
    xyz = np.array([
        [0.5, 1.5, 1.0],  # cell (0, 0): only one
        [1.5, 1.5, 5.0],  # cell (0, 1): higher
        [1.5, 1.5, 3.0],  # cell (0, 1): lower → drop
        [0.5, 0.5, 2.0],  # cell (1, 0)
        [1.5, 0.5, 7.0],  # cell (1, 1): higher
        [1.5, 0.5, 4.0],  # cell (1, 1): lower → drop
    ], dtype=np.float64)
    keep = decimate_highest(xyz, layout_2x2)
    np.testing.assert_array_equal(keep, [True, True, False, True, True, False])


def test_degenerate_single(layout_2x2):
    xyz = np.array([[0.5, 1.5, 1.0]], dtype=np.float64)
    keep = decimate_highest(xyz, layout_2x2)
    np.testing.assert_array_equal(keep, [True])


def test_degenerate_empty(layout_2x2):
    xyz = np.zeros((0, 3), dtype=np.float64)
    keep = decimate_highest(xyz, layout_2x2)
    assert keep.shape == (0,)
    assert keep.dtype == np.bool_


def test_corner_tie_first_wins(layout_2x2):
    """Strict < tie-break per src/LAS.cpp:613: only replace when zref < z."""
    xyz = np.array([
        [0.5, 0.5, 5.0],  # first in cell (1, 0)
        [0.5, 0.5, 5.0],  # equal z → first wins
    ], dtype=np.float64)
    keep = decimate_highest(xyz, layout_2x2)
    np.testing.assert_array_equal(keep, [True, False])


def test_corner_outside_grid_dropped(layout_2x2):
    xyz = np.array([
        [0.5, 1.5, 1.0],   # in
        [-1.0, 0.5, 99.0], # out (x < 0)
        [5.0, 0.5, 99.0],  # out (x >= ncol)
        [0.5, 5.0, 99.0],  # out (y > ymax)
    ], dtype=np.float64)
    keep = decimate_highest(xyz, layout_2x2)
    np.testing.assert_array_equal(keep, [True, False, False, False])


def test_dtype_rejected():
    layout = RasterLayout(xmin=0.0, ymax=1.0, xres=1.0, yres=1.0, ncol=1, nrow=1)
    xyz = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        decimate_highest(xyz, layout)


def test_shape_rejected():
    layout = RasterLayout(xmin=0.0, ymax=1.0, xres=1.0, yres=1.0, ncol=1, nrow=1)
    xyz = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        decimate_highest(xyz, layout)
