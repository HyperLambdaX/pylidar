"""chm_smooth — point-cloud height smoothing (lidR z_smooth port).

Each parametrized case loads ``tests/fixtures/chm_smooth_*.npz`` and
verifies the smoothed z column against the hand-derived expected output
within ``np.allclose(rtol=1e-6, atol=0)`` (spec §10.4).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.segmentation import chm_smooth


@pytest.mark.parametrize(
    "fixture_name",
    [
        "chm_smooth_happy",
        "chm_smooth_degenerate_single_point",
        "chm_smooth_corner_gaussian",
        "chm_smooth_corner_square_shape",
        "chm_smooth_corner_circular_three_neighbors",  # M3 review MED-3
        "chm_smooth_corner_nan_propagates",            # M3 review MED-4
    ],
)
def test_chm_smooth_matches_fixture(fixture_name, load_fixture):
    fx = load_fixture(fixture_name)
    z = chm_smooth(
        xyz=fx["inputs/xyz"],
        size=float(fx["inputs/size"]),
        method=str(fx["inputs/method"]),
        shape=str(fx["inputs/shape"]),
        sigma=float(fx["inputs/sigma"]),
    )
    assert z.dtype == np.float64
    assert z.shape == fx["expected/z"].shape
    np.testing.assert_allclose(z, fx["expected/z"], rtol=1e-6, atol=0)


def test_chm_smooth_default_sigma_is_size_over_six():
    """Spec §3 omits sigma; the wrapper defaults it to ``size/6`` to match
    lidR's R-side default (``smooth_height.R:34``). For ``method='average'``
    this default is irrelevant (sigma is unused), but we still verify the
    plumbing — gaussian with the default should equal gaussian with the
    same value passed explicitly."""
    xyz = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 8.0]], dtype=np.float64)
    z_default = chm_smooth(xyz=xyz, size=4.0, method="gaussian")
    z_explicit = chm_smooth(xyz=xyz, size=4.0, method="gaussian", sigma=4.0 / 6.0)
    np.testing.assert_array_equal(z_default, z_explicit)


def test_chm_smooth_rejects_wrong_dtype():
    xyz_f32 = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        chm_smooth(xyz=xyz_f32, size=2.0)


def test_chm_smooth_rejects_wrong_shape():
    xyz = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        chm_smooth(xyz=xyz, size=2.0)


def test_chm_smooth_rejects_invalid_size():
    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        chm_smooth(xyz=xyz, size=0.0)


def test_chm_smooth_rejects_invalid_method():
    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        chm_smooth(xyz=xyz, size=2.0, method="median")


def test_chm_smooth_rejects_invalid_shape_keyword():
    xyz = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        chm_smooth(xyz=xyz, size=2.0, shape="oblique")


def test_chm_smooth_rejects_non_contiguous():
    xyz = np.zeros((6, 3), dtype=np.float64)[::2]  # strided view
    with pytest.raises(ValueError):
        chm_smooth(xyz=xyz, size=2.0)
