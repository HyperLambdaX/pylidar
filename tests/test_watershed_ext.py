"""watershed_ext — direct EBImage-parity kernel tests (Phase 4.5).

Inline-data tests for ``pylidar._core.watershed_ext`` and the public
``pylidar.segmentation.watershed`` wrapper after the Phase 4.5 swap from
the skimage approximation to a 1:1 C++ port of EBImage's watershed.

Why no .npz fixtures
--------------------
Per the Phase 4.5 plan we'd compare bit-by-bit against EBImage R-side
output. The host has no R / EBImage env (and cannot install one), so we
fall back to hand-traced expectations derived from the C source itself
(EBImage src/watershed.cpp). PORT NOTE in
``src/core/its/watershed_ext.cpp`` documents the divergence on plateaus
(stable_sort vs R's non-stable rsort_with_index); the cases below are
authored to be plateau-free or rotationally-symmetric so the determinism
gap is invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar import _core
from pylidar.segmentation import watershed


# ─────────────────────────────────────────────────── core kernel: happy ──

def test_kernel_two_isolated_peaks():
    """Two clearly separated peaks → two distinct labels, valley = 0."""
    chm = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 5, 1, 0, 1, 5, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float64)
    out = _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)
    assert out.dtype == np.int32
    assert out.shape == chm.shape
    # Cell at (1, 3) is value 0 → background → label 0.
    assert out[1, 3] == 0
    # Two distinct labels overall.
    nonzero = out[out > 0]
    assert set(np.unique(nonzero).tolist()) == {1, 2}
    # The columns left of the valley share one label, columns right share
    # the other (peak labels are assigned in column-major *highest-first*
    # priority; we don't pin which side is "1" vs "2", just disjoint).
    left  = out[:, :3]
    right = out[:, 4:]
    assert set(np.unique(left[left > 0]).tolist()) | set(np.unique(right[right > 0]).tolist()) == {1, 2}
    assert len(np.unique(left[left > 0])) == 1
    assert len(np.unique(right[right > 0])) == 1


def test_kernel_single_peak():
    """One isolated peak with above-zero halo → all non-zero cells get 1."""
    chm = np.array([
        [0, 1, 1, 1, 0],
        [1, 2, 3, 2, 1],
        [1, 3, 5, 3, 1],
        [1, 2, 3, 2, 1],
        [0, 1, 1, 1, 0],
    ], dtype=np.float64)
    out = _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)
    # Every above-zero cell should receive label 1 (one peak).
    assert (out[chm > 0] == 1).all()
    assert (out[chm == 0] == 0).all()


# ─────────────────────────────────────────────────── tolerance behaviour ──

def test_kernel_tolerance_merges_shallow_hill():
    """Two adjacent peaks with shallow drop merge at high tolerance."""
    # Heights: 5, 1, 4 in a row — drop from 5→1 = 4, from 4→1 = 3. With
    # tolerance=10 both hills merge into one label; with tolerance=1 both
    # peaks stay separate.
    chm = np.array([
        [5, 4, 3, 2, 1, 2, 3, 4, 5],
    ], dtype=np.float64)

    merged   = _core.watershed_ext(chm=chm, tolerance=10.0, ext=1)
    distinct = _core.watershed_ext(chm=chm, tolerance=1.0,  ext=1)

    # tolerance=10 → drops are < tolerance → second peak gets merged into
    # the first → only one label remains.
    assert len(np.unique(merged[merged > 0])) == 1

    # tolerance=1 → both 5-peaks survive → two labels.
    assert len(np.unique(distinct[distinct > 0])) == 2


def test_kernel_tolerance_zero_keeps_every_max():
    """tolerance=0 keeps every regional maximum (lidR/EBImage allow it)."""
    chm = np.array([
        [0, 0, 0, 0, 0],
        [0, 5, 0, 5, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float64)
    out = _core.watershed_ext(chm=chm, tolerance=0.0, ext=1)
    # Two distinct peaks fully isolated by zeros → two labels.
    assert set(np.unique(out[out > 0]).tolist()) == {1, 2}


# ─────────────────────────────────────────────────── ext sweep ──

def test_kernel_ext_enlarges_neighbourhood():
    """A zero-gap between two peaks blocks ext=1 reachability but ext=2
    can bridge it, so a high `tolerance` only merges under ext=2.

    Layout ``[5 1 0 1 5]``:
      - ext=1: gap pixel at col 2 is value 0 (background) → it's never
        assigned, so nb sets at cols 1 and 3 each see only one peak →
        two seeds survive regardless of tolerance.
      - ext=2: col 1 reaches col 3, col 3 reaches col 1 — and once both
        cells are flooded by their respective peaks, the col-1 / col-3
        nb sets at later steps can include both seeds. With
        tolerance=10 (≥ drop=4) the two seeds merge into one.
    """
    chm_row = np.array([
        [5, 1, 0, 1, 5],
    ], dtype=np.float64)

    # tolerance=1 (< drop=4): both peaks always survive — every neighbour
    # of every gap pixel has diff ≥ tolerance, so check_multiple's merge
    # `if (diff >= tolerance) continue;` kicks them apart.
    out_ext1_strict = _core.watershed_ext(chm=chm_row, tolerance=1.0, ext=1)
    out_ext2_strict = _core.watershed_ext(chm=chm_row, tolerance=1.0, ext=2)
    assert set(np.unique(out_ext1_strict[out_ext1_strict > 0]).tolist()) == {1, 2}
    assert set(np.unique(out_ext2_strict[out_ext2_strict > 0]).tolist()) == {1, 2}

    # tolerance=10 (> drop=4): merging eligible. ext=1 still keeps two
    # because the zero gap blocks reachability; ext=2 bridges the gap and
    # the second peak gets folded into the first.
    out_ext1_loose = _core.watershed_ext(chm=chm_row, tolerance=10.0, ext=1)
    out_ext2_loose = _core.watershed_ext(chm=chm_row, tolerance=10.0, ext=2)
    assert len(np.unique(out_ext1_loose[out_ext1_loose > 0])) == 2  # zero-gap blocks ext=1
    assert len(np.unique(out_ext2_loose[out_ext2_loose > 0])) == 1  # ext=2 bridges + merges


# ─────────────────────────────────────────────────── degenerate inputs ──

def test_kernel_all_zero_no_labels():
    chm = np.zeros((4, 4), dtype=np.float64)
    out = _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)
    assert (out == 0).all()


def test_kernel_all_negative_no_labels():
    """Per the EBImage main loop guard `src[index[i]] > BG`, anything <= 0
    receives label 0."""
    chm = np.full((4, 4), -3.5, dtype=np.float64)
    out = _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)
    assert (out == 0).all()


def test_kernel_single_cell_peak():
    chm = np.array([[7.0]], dtype=np.float64)
    out = _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)
    assert out.shape == (1, 1)
    assert out[0, 0] == 1


def test_kernel_compaction_after_merges():
    """Even when multiple peaks merge mid-flow, output labels are
    contiguous 1..K (no gaps)."""
    # Three peaks of height 3 in a row, separated by 2-cell gaps of 1.
    # tolerance=10 forces them all to merge into one. Output should have
    # exactly one label = 1, never label 3 surviving alone.
    chm = np.array([
        [3, 1, 1, 3, 1, 1, 3],
    ], dtype=np.float64)
    out = _core.watershed_ext(chm=chm, tolerance=10.0, ext=2)
    # All non-zero cells share a single label that is exactly 1.
    assert (out[chm > 0] == 1).all()
    # No gap labels (e.g. label 2 or 3 must not appear).
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_kernel_determinism_on_plateau():
    """Repeated calls produce identical output (we use std::stable_sort
    instead of R's non-stable rsort_with_index — see PORT NOTE)."""
    rng = np.random.default_rng(123)
    chm = rng.integers(0, 3, size=(8, 8)).astype(np.float64)
    out1 = _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)
    out2 = _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)
    np.testing.assert_array_equal(out1, out2)


# ─────────────────────────────────────────────────── validation ──

def test_kernel_rejects_negative_tolerance():
    chm = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="tolerance"):
        _core.watershed_ext(chm=chm, tolerance=-0.1, ext=1)


def test_kernel_rejects_ext_zero():
    chm = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="ext"):
        _core.watershed_ext(chm=chm, tolerance=1.0, ext=0)


def test_kernel_rejects_wrong_dtype():
    chm = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        _core.watershed_ext(chm=chm, tolerance=1.0, ext=1)


# ─────────────────────────────────────────────────── public wrapper: ext ──

def test_public_watershed_ext_kw_passthrough():
    """The Phase 4.5 wrapper exposes `ext`; it forwards to _core.watershed_ext.

    Same zero-gap layout as test_kernel_ext_enlarges_neighbourhood, but
    th_tree=0.5 so the value-1 cells aren't masked out by the wrapper's
    `Canopy[Canopy < th_tree] <- 0` step.
    """
    chm = np.array([
        [5, 1, 0, 1, 5],
    ], dtype=np.float64)
    out_ext1 = watershed(chm=chm, th_tree=0.5, tol=10.0, ext=1)
    out_ext2 = watershed(chm=chm, th_tree=0.5, tol=10.0, ext=2)
    assert len(np.unique(out_ext1[out_ext1 > 0])) == 2
    assert len(np.unique(out_ext2[out_ext2 > 0])) == 1


def test_public_watershed_default_ext_is_one():
    """Default `ext` mirrors lidR's `watershed(..., ext = 1)`."""
    chm = np.array([
        [0, 5, 0, 5, 0],
    ], dtype=np.float64)
    out_default = watershed(chm=chm, th_tree=2.0, tol=1.0)
    out_explicit = watershed(chm=chm, th_tree=2.0, tol=1.0, ext=1)
    np.testing.assert_array_equal(out_default, out_explicit)


def test_public_watershed_rejects_ext_below_one():
    chm = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="ext"):
        watershed(chm=chm, ext=0)
