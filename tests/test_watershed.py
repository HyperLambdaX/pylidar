"""watershed — EBImage-port crown segmentation on a CHM.

Each parametrized case loads ``tests/fixtures/watershed_*.npz`` and
verifies the (H, W) int32 label output against hand-derived expected
labels (``np.array_equal`` per spec §10.4 — integer outputs require
strict equality).

History: these fixtures were originally authored in M3 against a
skimage approximation (see ``docs/port-notes.md`` §5 historical note).
Phase 4.5 (2026-05-13) replaced the runtime backend with a 1:1 C++
port of EBImage's ``src/watershed.cpp`` (``_core.watershed_ext``);
the fixture topologies are unambiguous enough that they continue to
pass under the new backend without any expected-value changes. New
EBImage-port-specific kernel tests live in
``tests/test_watershed_ext.py`` (Phase 4.5).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.segmentation import watershed


@pytest.mark.parametrize(
    "fixture_name",
    [
        "watershed_happy",
        "watershed_degenerate_below_th_tree",
        "watershed_corner_tol_gates_low_peak",
        "watershed_corner_nan_treated_as_no_tree",
    ],
)
def test_watershed_matches_fixture(fixture_name, load_fixture):
    fx = load_fixture(fixture_name)
    labels = watershed(
        chm=fx["inputs/chm"],
        th_tree=float(fx["inputs/th_tree"]),
        tol=float(fx["inputs/tol"]),
    )
    assert labels.dtype == np.int32
    assert labels.shape == fx["expected/labels"].shape
    np.testing.assert_array_equal(labels, fx["expected/labels"])


def test_watershed_rejects_wrong_dtype():
    chm_f32 = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        watershed(chm=chm_f32)


def test_watershed_rejects_wrong_ndim():
    chm = np.zeros((3,), dtype=np.float64)
    with pytest.raises(ValueError):
        watershed(chm=chm)


def test_watershed_rejects_negative_tol():
    chm = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        watershed(chm=chm, tol=-1.0)


def test_watershed_accepts_tol_zero():
    """lidR's `assert_is_a_number(tol)` (algorithm-its.R:331) does not
    require tol > 0; EBImage `?watershed` accepts tol >= 0. Match lidR by
    allowing tol=0, which keeps every regional maximum (h_maxima returns
    all maxima with dynamic >= 0 — i.e. all of them)."""
    chm = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)
    labels = watershed(chm=chm, th_tree=2.0, tol=0.0)
    # Single isolated peak ⇒ one tree.
    assert labels.dtype == np.int32
    assert labels[1, 1] == 1
    assert labels.sum() == 1


def test_watershed_rejects_non_contiguous():
    chm = np.zeros((6, 3), dtype=np.float64)[::2]  # strided view
    with pytest.raises(ValueError):
        watershed(chm=chm)
