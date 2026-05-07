"""lmf — both `lmf_points` (point-cloud kdtree path) and `lmf_chm` (raster
path). Each parametrized case loads `tests/fixtures/lmf_*.npz` and verifies
integer/bool equality against hand-derived expected output.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.segmentation import lmf_chm, lmf_points


# ---------------- lmf_points ----------------

@pytest.mark.parametrize(
    "fixture_name",
    [
        "lmf_points_happy",
        "lmf_points_degenerate_below_hmin",
        "lmf_points_corner_tie_equal_z",
        "lmf_points_corner_square_shape",                  # M2 review MEDIUM
        "lmf_points_corner_nonuniform_ws_cascade_off",     # M2 review MEDIUM
    ],
)
def test_lmf_points_matches_fixture(fixture_name, load_fixture):
    fx = load_fixture(fixture_name)
    # ws may be scalar (0-D ndarray) or per-point (1-D ndarray); pass it
    # through as-is so the wrapper exercises both paths.
    ws_in = fx["inputs/ws"]
    if ws_in.ndim == 0:
        ws_arg = float(ws_in)
    else:
        ws_arg = ws_in.astype(np.float64, copy=False)
    lm = lmf_points(
        xyz=fx["inputs/xyz"],
        ws=ws_arg,
        hmin=float(fx["inputs/hmin"]),
        shape=str(fx["inputs/shape"]),
    )
    assert lm.dtype == np.bool_
    assert lm.shape == fx["expected/lm"].shape
    np.testing.assert_array_equal(lm, fx["expected/lm"])


def test_lmf_points_callable_ws(load_fixture):
    """Callable ws must be called once per z and expanded by the Python
    wrapper into a per-point array. Spec §3 says callable(z)->float."""
    fx = load_fixture("lmf_points_happy")
    # Constant scalar function — should give same result as scalar ws=2.5.
    lm = lmf_points(
        xyz=fx["inputs/xyz"],
        ws=lambda z: float(z) * 0.0 + 2.5,
        hmin=float(fx["inputs/hmin"]),
        shape=str(fx["inputs/shape"]),
    )
    # callable path takes is_uniform=false (cascading off); for this
    # well-separated fixture both modes give the same lm output.
    np.testing.assert_array_equal(lm, fx["expected/lm"])


def test_lmf_points_rejects_callable_returning_array(load_fixture):
    fx = load_fixture("lmf_points_happy")
    with pytest.raises(ValueError, match=r"scalar"):
        lmf_points(
            xyz=fx["inputs/xyz"],
            ws=lambda z: np.array([2.5], dtype=np.float64),
            hmin=float(fx["inputs/hmin"]),
            shape=str(fx["inputs/shape"]),
        )


def test_lmf_points_array_ws(load_fixture):
    """Array ws path also goes through the wrapper; non-uniform but all-
    equal here so the LM verdict matches the scalar version."""
    fx = load_fixture("lmf_points_happy")
    n = fx["inputs/xyz"].shape[0]
    ws_arr = np.full((n,), 2.5, dtype=np.float64)
    lm = lmf_points(
        xyz=fx["inputs/xyz"],
        ws=ws_arr,
        hmin=float(fx["inputs/hmin"]),
        shape=str(fx["inputs/shape"]),
    )
    np.testing.assert_array_equal(lm, fx["expected/lm"])


def test_lmf_points_rejects_wrong_dtype(load_fixture):
    fx = load_fixture("lmf_points_happy")
    xyz_f32 = fx["inputs/xyz"].astype(np.float32)
    with pytest.raises(TypeError):
        lmf_points(xyz=xyz_f32, ws=2.5)


def test_lmf_points_rejects_invalid_shape():
    xyz = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        lmf_points(xyz=xyz, ws=2.5, shape="oblique")


def test_lmf_points_rejects_wrong_ws_length():
    xyz = np.zeros((3, 3), dtype=np.float64)
    bad_ws = np.array([1.0, 2.0], dtype=np.float64)  # length 2, expects 3
    with pytest.raises(ValueError):
        lmf_points(xyz=xyz, ws=bad_ws)


# ---------------- lmf_chm ----------------

@pytest.mark.parametrize(
    "fixture_name",
    [
        "lmf_chm_happy",
        "lmf_chm_degenerate_all_below_hmin",
        "lmf_chm_corner_tie_equal_neighbors",
        "lmf_chm_corner_square_shape",     # M2 review MEDIUM
    ],
)
def test_lmf_chm_matches_fixture(fixture_name, load_fixture):
    fx = load_fixture(fixture_name)
    coords = lmf_chm(
        chm=fx["inputs/chm"],
        ws=float(fx["inputs/ws"]),
        hmin=float(fx["inputs/hmin"]),
        shape=str(fx["inputs/shape"]),
    )
    assert coords.dtype == np.int32
    assert coords.shape[1] == 2
    np.testing.assert_array_equal(coords, fx["expected/coords"])


def test_lmf_chm_rejects_wrong_dtype():
    chm_f32 = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        lmf_chm(chm=chm_f32, ws=3.0)


def test_lmf_chm_rejects_invalid_ws():
    chm = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        lmf_chm(chm=chm, ws=0.0)


def test_lmf_chm_nan_treated_as_nodata():
    """NaN cells should never appear as LMs and never count as neighbours."""
    chm = np.array([
        [np.nan, 0.0, 0.0],
        [0.0,    5.0, 0.0],
        [0.0,    0.0, 0.0],
    ], dtype=np.float64)
    coords = lmf_chm(chm=chm, ws=3.0, hmin=2.0, shape="circular")
    np.testing.assert_array_equal(coords, np.array([[1, 1]], dtype=np.int32))
