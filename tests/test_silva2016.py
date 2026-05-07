"""silva2016 — happy / 退化 / corner per spec §6.

Spec §10 requires strict equality on integer label outputs (tree ids).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.segmentation import silva2016


def _run(fx: dict) -> np.ndarray:
    return silva2016(
        xyz=fx["inputs/xyz"],
        treetops=fx["inputs/treetops"],
        max_cr_factor=float(fx["inputs/max_cr_factor"]),
        exclusion=float(fx["inputs/exclusion"]),
    )


@pytest.mark.parametrize(
    "name",
    [
        "silva2016_happy",
        "silva2016_degenerate_empty",
        "silva2016_corner_boundary",
    ],
)
def test_silva2016_matches_fixture(load_fixture, name):
    fx = load_fixture(name)
    out = _run(fx)
    assert out.dtype == np.int32
    assert out.shape == fx["expected/ids"].shape
    np.testing.assert_array_equal(out, fx["expected/ids"])


def test_silva2016_rejects_wrong_dtype():
    treetops = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(TypeError, match=r"float64"):
        silva2016(xyz=np.zeros((5, 3), dtype=np.float32), treetops=treetops)
    xyz = np.zeros((5, 3), dtype=np.float64)
    with pytest.raises(TypeError, match=r"float64"):
        silva2016(xyz=xyz, treetops=np.zeros((1, 3), dtype=np.float32))


def test_silva2016_rejects_non_contiguous():
    base = np.zeros((5, 6), dtype=np.float64)
    xyz = base[:, :3]  # row-strided slice → non-c-contig view
    treetops = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"C-contiguous"):
        silva2016(xyz=xyz, treetops=treetops)


def test_silva2016_rejects_xyz_shape():
    xyz = np.zeros((5, 2), dtype=np.float64)
    treetops = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"xyz must"):
        silva2016(xyz=xyz, treetops=treetops)


def test_silva2016_rejects_treetops_shape():
    xyz = np.zeros((5, 3), dtype=np.float64)
    treetops = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"treetops must"):
        silva2016(xyz=xyz, treetops=treetops)


def test_silva2016_rejects_exclusion_at_boundary():
    xyz = np.zeros((1, 3), dtype=np.float64)
    treetops = np.zeros((1, 3), dtype=np.float64)
    # exclusion in (0, 1) — open interval — boundary should fail.
    with pytest.raises(ValueError, match=r"exclusion"):
        silva2016(xyz=xyz, treetops=treetops, exclusion=0.0)
    with pytest.raises(ValueError, match=r"exclusion"):
        silva2016(xyz=xyz, treetops=treetops, exclusion=1.0)


def test_silva2016_rejects_negative_max_cr_factor():
    xyz = np.zeros((1, 3), dtype=np.float64)
    treetops = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"max_cr_factor"):
        silva2016(xyz=xyz, treetops=treetops, max_cr_factor=0.0)
