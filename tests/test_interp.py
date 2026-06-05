"""Generic scattered interpolation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pylidar._interp import knnidw_interpolate, tin_interpolate


def test_tin_interpolate_inside_hull_linear_plane():
    src_xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    src_z = src_xy[:, 0] + 2.0 * src_xy[:, 1]
    query_xy = np.array([[0.25, 0.5]], dtype=np.float64)

    out = tin_interpolate(src_xy, src_z, query_xy, extrapolate=None)

    assert out == pytest.approx([1.25])


def test_tin_interpolate_outside_hull_without_extrapolate_is_nan():
    src_xy = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    src_z = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    query_xy = np.array([[2.0, 2.0]], dtype=np.float64)

    out = tin_interpolate(src_xy, src_z, query_xy, extrapolate=None)

    assert np.isnan(out[0])


def test_tin_interpolate_outside_hull_uses_knnidw_extrapolate():
    src_xy = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    src_z = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    query_xy = np.array([[2.0, 0.0]], dtype=np.float64)

    out = tin_interpolate(
        src_xy,
        src_z,
        query_xy,
        extrapolate="knnidw",
        extrapolate_k=1,
        extrapolate_p=1.0,
        extrapolate_rmax=5.0,
    )

    assert out == pytest.approx([10.0])


def test_knnidw_interpolate_exact_hit_returns_source_value():
    src_xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    src_z = np.array([4.0, 10.0], dtype=np.float64)
    query_xy = np.array([[0.0, 0.0]], dtype=np.float64)

    out = knnidw_interpolate(src_xy, src_z, query_xy, k=2, p=2.0)

    assert out == pytest.approx([4.0])


def test_knnidw_interpolate_rmax_without_neighbours_returns_nan():
    src_xy = np.array([[0.0, 0.0]], dtype=np.float64)
    src_z = np.array([4.0], dtype=np.float64)
    query_xy = np.array([[10.0, 0.0]], dtype=np.float64)

    out = knnidw_interpolate(src_xy, src_z, query_xy, k=1, p=2.0, rmax=1.0)

    assert np.isnan(out[0])


def test_interpolate_rejects_bad_shapes():
    with pytest.raises(ValueError, match="src_xy"):
        tin_interpolate(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([[0.0, 0.0]], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="src_z length"):
        knnidw_interpolate(
            np.array([[0.0, 0.0]], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[0.0, 0.0]], dtype=np.float64),
        )
