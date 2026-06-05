"""PMF ground-classification kernel ported from lidR."""

from __future__ import annotations

import numpy as np
import pytest

from pylidar import _core


def _grid3(z_center: float) -> np.ndarray:
    pts = []
    for y in range(3):
        for x in range(3):
            z = z_center if (x == 1 and y == 1) else 0.0
            pts.append((float(x), float(y), z))
    return np.asarray(pts, dtype=np.float64)


def _pmf(xyz, *, ws=(3.0,), th=(1.0,), candidate_mask=None):
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    if candidate_mask is None:
        candidate_mask = np.ones(xyz.shape[0], dtype=np.bool_)
    return _core.pmf_ground(
        xyz=xyz,
        ws=np.ascontiguousarray(ws, dtype=np.float64),
        th=np.ascontiguousarray(th, dtype=np.float64),
        candidate_mask=np.ascontiguousarray(candidate_mask, dtype=np.bool_),
    )


def test_pmf_flat_plane_stays_ground():
    xyz = _grid3(0.0)

    out = _pmf(xyz, ws=(3.0, 5.0), th=(0.5, 0.5))

    assert out.dtype == np.bool_
    assert out.all()


def test_pmf_tall_narrow_bump_is_removed():
    xyz = _grid3(10.0)

    out = _pmf(xyz, ws=(3.0,), th=(1.0,))

    assert out.reshape(3, 3).tolist() == [
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ]


def test_pmf_threshold_boundary_drops_when_difference_equals_threshold():
    xyz = _grid3(1.0)

    out = _pmf(xyz, ws=(3.0,), th=(1.0,))

    assert not bool(out.reshape(3, 3)[1, 1])


def test_pmf_candidate_mask_limits_neighbours_and_output():
    xyz = _grid3(10.0)
    candidate = np.ones(xyz.shape[0], dtype=np.bool_)
    candidate[4] = False

    out = _pmf(xyz, ws=(3.0,), th=(1.0,), candidate_mask=candidate)

    assert not bool(out[4])
    assert out.sum() == 8


def test_pmf_replicates_lidr_negative_z_opening_quirk():
    xyz = _grid3(-1.0)
    xyz[:, 2] = -5.0
    xyz[4, 2] = -1.0

    out = _pmf(xyz, ws=(3.0,), th=(1.0,))

    # lidR pass 2 initializes max with numeric_limits<double>::min(), not
    # lowest(), so all-negative pass-1 windows open to a tiny positive value.
    # The negative bump is therefore kept by the progressive threshold test.
    assert out.all()


def test_pmf_binding_rejects_bad_inputs():
    xyz = _grid3(0.0)
    candidate = np.ones(xyz.shape[0], dtype=np.bool_)

    with pytest.raises(ValueError, match="ws and th length"):
        _core.pmf_ground(
            xyz=xyz,
            ws=np.array([3.0], dtype=np.float64),
            th=np.array([1.0, 2.0], dtype=np.float64),
            candidate_mask=candidate,
        )
    with pytest.raises(ValueError, match="candidate_mask length"):
        _core.pmf_ground(
            xyz=xyz,
            ws=np.array([3.0], dtype=np.float64),
            th=np.array([1.0], dtype=np.float64),
            candidate_mask=np.ones(2, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="xyz must be finite"):
        bad = xyz.copy()
        bad[0, 2] = np.nan
        _pmf(bad)
    with pytest.raises(ValueError, match="ws must be finite and > 0"):
        _pmf(xyz, ws=(0.0,))
    with pytest.raises(ValueError, match="th must be finite and > 0"):
        _pmf(xyz, th=(np.nan,))
