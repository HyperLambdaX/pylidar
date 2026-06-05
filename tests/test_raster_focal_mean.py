"""focal_mean — lidR/terra-style CHM focal mean helper."""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.raster import focal_mean


def test_focal_mean_3x3_ignores_nan():
    chm = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    out = focal_mean(chm)
    assert out[1, 1] == 5.0  # sum 40 / 8 finite cells


def test_focal_mean_edges_treat_off_raster_as_nan():
    chm = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    out = focal_mean(chm)
    assert out[0, 0] == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)


def test_focal_mean_all_nan_window_stays_nan():
    chm = np.full((3, 3), np.nan, dtype=np.float64)
    out = focal_mean(chm)
    assert np.isnan(out).all()


def test_focal_mean_window_size_1_is_identity_for_finite_values():
    chm = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
    out = focal_mean(chm, window_size=1)
    np.testing.assert_array_equal(out, chm)


def test_focal_mean_rejects_bad_inputs():
    with pytest.raises(TypeError):
        focal_mean(np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        focal_mean(np.zeros((3,), dtype=np.float64))
    with pytest.raises(ValueError):
        focal_mean(np.zeros((3, 3), dtype=np.float64), window_size=2)
