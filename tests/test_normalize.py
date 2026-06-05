"""Height normalization API."""

from __future__ import annotations

import numpy as np
import pytest

import pylidar
from pylidar.normalize import normalize_height, normalize_height_with_metadata


def test_normalize_height_tin_on_plane():
    xyz = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 11.0],
            [0.0, 1.0, 12.0],
            [0.25, 0.25, 13.75],
        ],
        dtype=np.float64,
    )
    ground = np.array([True, True, True, False], dtype=np.bool_)

    z = normalize_height(xyz=xyz, ground_mask=ground)

    assert z == pytest.approx([0.0, 0.0, 0.0, 3.0])


def test_normalize_height_tin_knnidw_extrapolates_outside_hull():
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [2.0, 0.0, 12.0],
        ],
        dtype=np.float64,
    )
    ground = np.array([True, True, True, False], dtype=np.bool_)

    result = normalize_height_with_metadata(xyz=xyz, ground_mask=ground)

    assert result.fallback_counts["tin_missing"] == 1
    assert result.fallback_counts["knnidw_extrapolated"] == 1
    assert result.fallback_counts["nearest_rescued"] == 0
    assert result.z[3] == pytest.approx(2.271086873171143)


def test_normalize_height_final_nearest_rescue_after_rmax_miss():
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [100.0, 0.0, 105.0],
        ],
        dtype=np.float64,
    )
    ground = np.array([True, True, True, False], dtype=np.bool_)

    result = normalize_height_with_metadata(xyz=xyz, ground_mask=ground)

    assert result.fallback_counts["tin_missing"] == 1
    assert result.fallback_counts["knnidw_extrapolated"] == 0
    assert result.fallback_counts["nearest_rescued"] == 1
    assert result.z[3] == pytest.approx(95.0)


def test_normalize_height_knnidw_method():
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [1.0, 0.0, 6.0],
        ],
        dtype=np.float64,
    )
    ground = np.array([True, True, False], dtype=np.bool_)

    z = normalize_height(xyz=xyz, ground_mask=ground, method="knnidw")

    assert z[2] == pytest.approx(5.0)


def test_normalize_top_level_export():
    assert pylidar.normalize_height is normalize_height
    assert pylidar.normalize.normalize_height is normalize_height


def test_normalize_height_validation():
    xyz = np.zeros((3, 3), dtype=np.float64)
    ground = np.array([True, False, False], dtype=np.bool_)

    with pytest.raises(ValueError, match="at least 3 ground"):
        normalize_height(xyz=xyz, ground_mask=ground)
    with pytest.raises(ValueError, match="contains no ground"):
        normalize_height(xyz=xyz, ground_mask=np.zeros(3, dtype=np.bool_))
    with pytest.raises(ValueError, match="ground_mask length"):
        normalize_height(xyz=xyz, ground_mask=np.ones(2, dtype=np.bool_))
    with pytest.raises(ValueError, match="method"):
        normalize_height(xyz=xyz, ground_mask=np.ones(3, dtype=np.bool_), method="bad")
