"""Public ground-classification API."""

from __future__ import annotations

import numpy as np
import pytest

import pylidar
from pylidar.ground import classify_ground, csf, pmf, util_makeZhangParam


def _grid3(z_center: float) -> np.ndarray:
    pts = []
    for y in range(3):
        for x in range(3):
            z = z_center if (x == 1 and y == 1) else 0.0
            pts.append((float(x), float(y), z))
    return np.asarray(pts, dtype=np.float64)


def _flat_ground_with_canopy() -> np.ndarray:
    rng = np.random.default_rng(0)
    ground = np.c_[
        rng.uniform(0.0, 20.0, 300),
        rng.uniform(0.0, 20.0, 300),
        rng.normal(0.0, 0.03, 300),
    ]
    canopy = np.c_[
        rng.uniform(8.0, 12.0, 80),
        rng.uniform(8.0, 12.0, 80),
        rng.uniform(4.0, 8.0, 80),
    ]
    return np.ascontiguousarray(np.r_[ground, canopy].astype(np.float64))


def test_util_make_zhang_param_linear_defaults():
    ws, th = util_makeZhangParam()

    assert ws.dtype == np.float64
    assert th.dtype == np.float64
    assert ws.tolist() == [5.0, 9.0, 13.0, 17.0]
    assert th.tolist() == [3.0, 3.0, 3.0, 3.0]


def test_util_make_zhang_param_exponential():
    ws, th = util_makeZhangParam(exp=True, max_ws=20.0)

    assert ws.tolist() == [3.0, 5.0, 9.0, 17.0]
    assert th.tolist() == [0.5, 2.5, 3.0, 3.0]


def test_util_make_zhang_param_validation_and_warning():
    with pytest.raises(ValueError, match="b cannot"):
        util_makeZhangParam(exp=True, b=1.0)
    with pytest.raises(ValueError, match="dh0 greater"):
        util_makeZhangParam(dh0=3.0, dhmax=3.0)
    with pytest.raises(ValueError, match="Minimum windows"):
        util_makeZhangParam(max_ws=2.0)
    with pytest.warns(RuntimeWarning, match="incoherence"):
        util_makeZhangParam(b=0.5)


def test_classify_ground_pmf_candidate_none_means_all_points():
    xyz = _grid3(10.0)
    alg = pmf(ws=np.array([3.0]), th=np.array([1.0]))

    out = classify_ground(xyz=xyz, algorithm=alg)

    assert out.reshape(3, 3).tolist() == [
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ]


def test_classify_ground_respects_candidate_mask():
    xyz = _grid3(10.0)
    candidate = np.ones(xyz.shape[0], dtype=np.bool_)
    candidate[4] = False
    alg = pmf(ws=np.array([3.0]), th=np.array([1.0]))

    out = classify_ground(xyz=xyz, algorithm=alg, candidate_mask=candidate)

    assert not bool(out[4])
    assert out.sum() == 8


def test_classify_ground_pmf_shifts_negative_z_before_core_quirk():
    xyz = _grid3(-1.0)
    xyz[:, 2] = -5.0
    xyz[4, 2] = -1.0
    alg = pmf(ws=np.array([3.0]), th=np.array([1.0]))

    out = classify_ground(xyz=xyz, algorithm=alg)

    assert out.reshape(3, 3).tolist() == [
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ]


def test_classify_ground_pmf_negative_z_matches_positive_shifted_result():
    rng = np.random.default_rng(123)
    n_ground = 400
    n_canopy = 40
    xg = rng.uniform(0.0, 40.0, n_ground)
    yg = rng.uniform(0.0, 40.0, n_ground)
    zg = -20.0 + 0.1 * xg + 0.05 * yg + rng.normal(0.0, 0.03, n_ground)
    xc = rng.uniform(16.0, 24.0, n_canopy)
    yc = rng.uniform(16.0, 24.0, n_canopy)
    zc_ground = -20.0 + 0.1 * xc + 0.05 * yc
    zc = zc_ground + rng.uniform(3.0, 6.0, n_canopy)
    xyz = np.c_[np.r_[xg, xc], np.r_[yg, yc], np.r_[zg, zc]].astype(np.float64)
    shifted = xyz.copy()
    shifted[:, 2] += -shifted[:, 2].min() + 1.0
    ws, th = util_makeZhangParam(max_ws=20.0)
    alg = pmf(ws=ws, th=th)

    out = classify_ground(xyz=xyz, algorithm=alg)
    expected = classify_ground(xyz=shifted, algorithm=alg)

    assert np.array_equal(out, expected)
    assert int(out[n_ground:].sum()) < n_canopy


def test_ground_top_level_export():
    assert pylidar.classify_ground is classify_ground
    assert pylidar.ground.pmf is pmf
    assert pylidar.ground.csf is csf


def test_classify_ground_csf_flat_ground_with_canopy():
    xyz = _flat_ground_with_canopy()

    out = classify_ground(xyz=xyz, algorithm=csf(iterations=100))

    assert out.dtype == np.bool_
    assert int(out[:300].sum()) >= 280
    assert not out[300:].any()


def test_classify_ground_csf_respects_candidate_mask():
    xyz = _flat_ground_with_canopy()
    candidate = np.ones(xyz.shape[0], dtype=np.bool_)
    candidate[:10] = False

    out = classify_ground(
        xyz=xyz,
        algorithm=csf(iterations=100),
        candidate_mask=candidate,
    )

    assert not out[:10].any()
    assert int(out[10:300].sum()) >= 270


def test_pmf_public_validation():
    with pytest.raises(ValueError, match="ws and th length"):
        pmf(ws=np.array([3.0]), th=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="ws must be finite"):
        pmf(ws=np.array([np.nan]), th=np.array([1.0]))
    with pytest.raises(ValueError, match="th must be finite"):
        pmf(ws=np.array([3.0]), th=np.array([0.0]))


def test_csf_public_validation():
    with pytest.raises(TypeError, match="slope_smooth"):
        csf(slope_smooth=1)
    with pytest.raises(ValueError, match="class_threshold"):
        csf(class_threshold=0.0)
    with pytest.raises(ValueError, match="cloth_resolution"):
        csf(cloth_resolution=np.nan)
    with pytest.raises(ValueError, match="rigidness"):
        csf(rigidness=4)
    with pytest.raises(ValueError, match="iterations"):
        csf(iterations=0)
    with pytest.raises(ValueError, match="time_step"):
        csf(time_step=-1.0)


def test_classify_ground_public_validation():
    alg = pmf(ws=np.array([3.0]), th=np.array([1.0]))
    with pytest.raises(ValueError, match="xyz must have shape"):
        classify_ground(xyz=np.zeros((3,), dtype=np.float64), algorithm=alg)
    with pytest.raises(ValueError, match="candidate_mask length"):
        classify_ground(
            xyz=np.zeros((3, 3), dtype=np.float64),
            algorithm=alg,
            candidate_mask=np.ones(2, dtype=np.bool_),
        )
    with pytest.raises(TypeError, match="unsupported ground algorithm"):
        classify_ground(
            xyz=np.zeros((3, 3), dtype=np.float64),
            algorithm=object(),
        )
