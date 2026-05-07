"""li2012 (Li, Guo, Jakubowski & Kelly 2012) — port behaviour vs hand-derived
fixtures.

Each parametrized case loads `tests/fixtures/li2012_*.npz` (see
`tests/fixtures/_make_fixtures.py` for the hand-trace) and checks integer-id
equality against the upstream-derived expected output.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.segmentation import li2012


@pytest.mark.parametrize(
    "fixture_name",
    [
        "li2012_happy",
        "li2012_degenerate_below_hmin",
        "li2012_corner_th_boundary",
        "li2012_corner_radius_clip",         # M2 review HIGH regression lock
        "li2012_corner_R0_skips_prepass",    # M2 review MEDIUM coverage gap
    ],
)
def test_matches_fixture(fixture_name, load_fixture):
    fx = load_fixture(fixture_name)
    ids = li2012(
        xyz=fx["inputs/xyz"],
        dt1=float(fx["inputs/dt1"]),
        dt2=float(fx["inputs/dt2"]),
        Zu=float(fx["inputs/Zu"]),
        R=float(fx["inputs/R"]),
        hmin=float(fx["inputs/hmin"]),
        speed_up=float(fx["inputs/speed_up"]),
    )
    assert ids.dtype == np.int32
    assert ids.shape == fx["expected/ids"].shape
    np.testing.assert_array_equal(ids, fx["expected/ids"])


def test_rejects_wrong_dtype(load_fixture):
    fx = load_fixture("li2012_happy")
    xyz_f32 = fx["inputs/xyz"].astype(np.float32)
    with pytest.raises(TypeError):
        li2012(xyz=xyz_f32)


def test_rejects_wrong_shape(load_fixture):
    fx = load_fixture("li2012_happy")
    bad_xyz = fx["inputs/xyz"][:, :2].copy()  # (N, 2) instead of (N, 3)
    with pytest.raises((ValueError, TypeError)):
        li2012(xyz=bad_xyz)


def test_rejects_invalid_dt():
    xyz = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        li2012(xyz=xyz, dt1=0.0)
    with pytest.raises(ValueError):
        li2012(xyz=xyz, dt2=-1.0)


def test_rejects_invalid_zu_and_hmin():
    xyz = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"Zu"):
        li2012(xyz=xyz, Zu=0.0)
    with pytest.raises(ValueError, match=r"hmin"):
        li2012(xyz=xyz, hmin=0.0)
