"""dalponte2016 — happy / 退化 / corner per spec §6.

Spec §10 requires strict equality on integer label outputs (region ids).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.segmentation import dalponte2016


def _run(fx: dict) -> np.ndarray:
    return dalponte2016(
        chm=fx["inputs/chm"],
        seeds=fx["inputs/seeds"],
        th_tree=float(fx["inputs/th_tree"]),
        th_seed=float(fx["inputs/th_seed"]),
        th_cr=float(fx["inputs/th_cr"]),
        max_cr=float(fx["inputs/max_cr"]),
    )


@pytest.mark.parametrize(
    "name",
    [
        "dalponte2016_happy",
        "dalponte2016_degenerate",
        "dalponte2016_corner_tie",
    ],
)
def test_dalponte2016_matches_fixture(load_fixture, name):
    fx = load_fixture(name)
    out = _run(fx)
    assert out.dtype == np.int32
    assert out.shape == fx["expected/regions"].shape
    np.testing.assert_array_equal(out, fx["expected/regions"])


def test_dalponte2016_rejects_shape_mismatch():
    chm = np.zeros((4, 4), dtype=np.float64)
    seeds = np.zeros((4, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="share shape"):
        dalponte2016(chm=chm, seeds=seeds)


def test_dalponte2016_rejects_th_seed_out_of_range():
    chm = np.zeros((4, 4), dtype=np.float64)
    seeds = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match=r"th_seed"):
        dalponte2016(chm=chm, seeds=seeds, th_seed=1.5)


def test_dalponte2016_rejects_wrong_dtype():
    chm = np.zeros((4, 4), dtype=np.float32)
    seeds = np.zeros((4, 4), dtype=np.int32)
    # nanobind raises TypeError for dtype mismatches.
    with pytest.raises(TypeError):
        dalponte2016(chm=chm, seeds=seeds)


def test_dalponte2016_threshold_boundaries_are_strict():
    seeds = np.zeros((5, 5), dtype=np.int32)
    seeds[2, 2] = 1

    base = np.zeros((5, 5), dtype=np.float64)
    base[2, 2] = 10.0

    # pz == th_tree fails because lidR uses `px.z > th_tree`.
    chm = base.copy()
    chm[2, 3] = 6.0
    out = dalponte2016(
        chm=chm,
        seeds=seeds,
        th_tree=6.0,
        th_seed=0.45,
        th_cr=0.55,
        max_cr=10.0,
    )
    np.testing.assert_array_equal(out, seeds)

    # pz == th_seed * hSeed fails because lidR uses strict `>`.
    chm = base.copy()
    chm[2, 3] = 4.5
    out = dalponte2016(
        chm=chm,
        seeds=seeds,
        th_tree=2.0,
        th_seed=0.45,
        th_cr=0.10,
        max_cr=10.0,
    )
    np.testing.assert_array_equal(out, seeds)

    # pz == th_cr * mean_crown_height fails because lidR uses strict `>`.
    chm = base.copy()
    chm[2, 3] = 5.5
    out = dalponte2016(
        chm=chm,
        seeds=seeds,
        th_tree=2.0,
        th_seed=0.10,
        th_cr=0.55,
        max_cr=10.0,
    )
    np.testing.assert_array_equal(out, seeds)

    # abs(delta) == max_cr fails because lidR uses strict `< DIST`.
    chm = base.copy()
    chm[2, 3] = 6.0
    out = dalponte2016(
        chm=chm,
        seeds=seeds,
        th_tree=2.0,
        th_seed=0.45,
        th_cr=0.55,
        max_cr=1.0,
    )
    np.testing.assert_array_equal(out, seeds)
