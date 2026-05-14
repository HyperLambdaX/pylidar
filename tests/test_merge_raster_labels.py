"""Tests for ``pylidar.merge.merge_raster_labels`` and the top-level
``pylidar.segment_trees`` dispatch."""

from __future__ import annotations

import numpy as np
import pytest

import pylidar
from pylidar.merge import merge_raster_labels
from pylidar.raster import RasterLayout


def _layout_3x3() -> RasterLayout:
    # xmin=0, ymax=3, 1m cells → covers x∈[0,3], y∈[0,3]
    return RasterLayout(xmin=0.0, ymax=3.0, xres=1.0, yres=1.0, ncol=3, nrow=3)


def test_merge_happy_path_in_grid_assigns_label_at_cell():
    layout = _layout_3x3()
    labels = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int32)
    xy = np.array(
        [
            [0.5, 2.5],  # cell (row=0, col=0) → 10
            [1.5, 1.5],  # cell (row=1, col=1) → 50
            [2.5, 0.5],  # cell (row=2, col=2) → 90
        ],
        dtype=np.float64,
    )
    out = merge_raster_labels(xy, layout, labels)
    assert out.dtype == np.int32
    assert list(out) == [10, 50, 90]


def test_merge_out_of_grid_points_default_int32_max_sentinel():
    """Phase 5 audit fix #3 (2026-05-12): default nodata is the LAS-spec
    NA sentinel matching ``labels.dtype`` (was 0 before the audit; that
    diverged from the writer's ExtraBytesVlr no_data field)."""
    layout = _layout_3x3()
    labels = np.full(layout.shape, 7, dtype=np.int32)
    xy = np.array([[-1.0, 1.5], [10.0, 1.5], [1.5, 10.0]], dtype=np.float64)
    out = merge_raster_labels(xy, layout, labels)
    int32_max = np.iinfo(np.int32).max
    assert list(out) == [int32_max, int32_max, int32_max]


def test_merge_out_of_grid_points_default_float64_tiny_sentinel():
    layout = _layout_3x3()
    labels = np.full(layout.shape, 7.5, dtype=np.float64)
    xy = np.array([[-1.0, 1.5]], dtype=np.float64)
    out = merge_raster_labels(xy, layout, labels)
    assert out[0] == np.finfo(np.float64).tiny


def test_merge_explicit_nodata_zero_opts_into_pylidar_convention():
    """Callers can opt back into the 0=no-tree convention for symmetry
    with the in-grid algorithm sentinel."""
    layout = _layout_3x3()
    labels = np.full(layout.shape, 7, dtype=np.int32)
    xy = np.array([[-1.0, 1.5]], dtype=np.float64)
    out = merge_raster_labels(xy, layout, labels, nodata=0)
    assert out[0] == 0


def test_merge_out_of_grid_points_respect_custom_nodata():
    layout = _layout_3x3()
    labels = np.full(layout.shape, 7, dtype=np.int32)
    xy = np.array([[1.5, 1.5], [-1.0, 1.5]], dtype=np.float64)
    out = merge_raster_labels(xy, layout, labels, nodata=-1)
    assert out[0] == 7
    assert out[1] == -1


def test_merge_dtype_propagates_from_labels():
    layout = _layout_3x3()
    labels = np.full(layout.shape, 1.25, dtype=np.float64)
    xy = np.array([[0.5, 0.5]], dtype=np.float64)
    out = merge_raster_labels(xy, layout, labels, nodata=0.0)
    assert out.dtype == np.float64
    assert out[0] == 1.25


def test_merge_empty_xy_returns_empty():
    layout = _layout_3x3()
    labels = np.zeros(layout.shape, dtype=np.int32)
    xy = np.empty((0, 2), dtype=np.float64)
    out = merge_raster_labels(xy, layout, labels)
    assert out.shape == (0,)
    assert out.dtype == np.int32


def test_merge_right_bottom_edge_points_clamp_into_grid():
    """xy on the xmax/ymin edges clamp inward per RasterLayout convention."""
    layout = _layout_3x3()
    labels = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int32)
    # x=3.0 is xmax → col=2; y=0.0 is ymin → row=2.
    xy = np.array([[3.0, 0.0]], dtype=np.float64)
    out = merge_raster_labels(xy, layout, labels)
    assert out[0] == 90


def test_merge_rejects_wrong_xy_dtype():
    layout = _layout_3x3()
    labels = np.zeros(layout.shape, dtype=np.int32)
    xy = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        merge_raster_labels(xy, layout, labels)


def test_merge_rejects_wrong_xy_shape():
    layout = _layout_3x3()
    labels = np.zeros(layout.shape, dtype=np.int32)
    xy = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        merge_raster_labels(xy, layout, labels)


def test_merge_rejects_labels_layout_shape_mismatch():
    layout = _layout_3x3()
    labels = np.zeros((4, 4), dtype=np.int32)
    xy = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="layout.shape"):
        merge_raster_labels(xy, layout, labels)


def test_merge_rejects_non_ndarray_labels():
    layout = _layout_3x3()
    xy = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(TypeError, match="ndarray"):
        merge_raster_labels(xy, layout, [[1, 2], [3, 4]])  # type: ignore[arg-type]


# ---------------------------------------------------------------- segment_trees


def test_segment_trees_raster_path_matches_merge():
    layout = _layout_3x3()
    labels = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int32)
    xyz = np.array(
        [[0.5, 2.5, 99.0], [1.5, 1.5, 50.0], [2.5, 0.5, 10.0]], dtype=np.float64
    )
    out = pylidar.segment_trees(xyz, layout, raster_labels=labels)
    assert list(out) == [1, 2, 3]


def test_segment_trees_point_path_copies_labels():
    xyz = np.zeros((4, 3), dtype=np.float64)
    point_labels = np.array([10, 20, 30, 40], dtype=np.int32)
    out = pylidar.segment_trees(xyz, point_labels=point_labels)
    assert list(out) == [10, 20, 30, 40]
    assert out is not point_labels  # returns a copy


def test_segment_trees_raster_requires_layout():
    xyz = np.zeros((1, 3), dtype=np.float64)
    labels = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="layout"):
        pylidar.segment_trees(xyz, None, raster_labels=labels)


def test_segment_trees_rejects_both_raster_and_point_labels():
    xyz = np.zeros((3, 3), dtype=np.float64)
    layout = _layout_3x3()
    labels = np.zeros((3, 3), dtype=np.int32)
    point_labels = np.zeros(3, dtype=np.int32)
    with pytest.raises(ValueError, match="exactly one"):
        pylidar.segment_trees(
            xyz, layout, raster_labels=labels, point_labels=point_labels
        )


def test_segment_trees_rejects_neither_label_supplied():
    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="exactly one"):
        pylidar.segment_trees(xyz)


def test_segment_trees_point_path_length_mismatch_raises():
    xyz = np.zeros((4, 3), dtype=np.float64)
    point_labels = np.zeros(3, dtype=np.int32)
    with pytest.raises(ValueError, match="length"):
        pylidar.segment_trees(xyz, point_labels=point_labels)
