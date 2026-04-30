"""Phase 4 acceptance tests — segment_silva2016.

Five task_plan-required cases:
  1. Silva ≥ Dalponte on the same CHM/seeds (Voronoi reaches every non-NaN
     cell; dalponte's region-grow is gated by `pz > h_seed*th_seed`).
  2. Large height-range CHM: exclusion=0.3 cuts low cells, leaving the
     inner-high-cells crown.
  3. max_cr_factor extremes (very large → only exclusion gates; very
     small → crown collapses to seed cell).
  4. Single seed → entire non-NaN raster forms one Voronoi cell.
  5. exclusion ∈ {0, 1} → ValueError (open interval; differs from
     dalponte's closed [0, 1]).

Plus extras: dtype validation, (M,3) auto-ID vs (M,4) custom ID, NaN
cells, 0 seeds, parametrised invalid scalars, C++ direct-call self-check,
and one xfail covering the Phase 3 D2 deferred fix (M,3 auto-ID order).
"""

from __future__ import annotations

import numpy as np
import pytest

import pylidar
from pylidar import _core


# ---------------------------------------------------------------------------
# task_plan #1 — silva covers ≥ as many pixels as dalponte on the same input.
#
# CHM: center z=10, surrounded by z=3 cells. With dalponte default
# (th_seed=0.45), neighbour z=3 fails `3 > 10*0.45 = 4.5` → crown stops at
# the seed (1 pixel). Silva sees one Voronoi group with hmax=10; both
# thresholds (`3 >= 0.3*10` and `dist <= 0.6*10`) pass for every pixel →
# crown = full 25 pixels.
# ---------------------------------------------------------------------------


def test_silva_more_inclusive_than_dalponte_on_low_skirt_chm():
    chm = np.full((5, 5), 3.0, dtype=np.float64)
    chm[2, 2] = 10.0
    transform = (0.0, 4.0, 1.0)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)

    crowns_silva = pylidar.segment_silva2016(chm, transform, seeds)
    crowns_dalponte = pylidar.segment_dalponte2016(chm, transform, seeds)

    silva_count = int((crowns_silva != 0).sum())
    dalponte_count = int((crowns_dalponte != 0).sum())
    assert silva_count >= dalponte_count
    assert silva_count == 25
    assert dalponte_count == 1


# ---------------------------------------------------------------------------
# task_plan #2 — exclusion cuts the low ring; inner cells survive.
# ---------------------------------------------------------------------------


def test_silva_exclusion_excises_low_ring():
    # 7x7 CHM: center=20, ring=10, outer=1. exclusion default 0.3 →
    # threshold = 6. Cells z=10 pass (10>=6); cells z=1 fail (1<6).
    chm = np.full((7, 7), 1.0, dtype=np.float64)
    chm[2:5, 2:5] = 10.0
    chm[3, 3]     = 20.0
    transform = (0.0, 6.0, 1.0)  # pixel (3,3) world center = (3.0, 3.0)
    seeds = np.array([[3.0, 3.0, 20.0]], dtype=np.float64)

    crowns = pylidar.segment_silva2016(chm, transform, seeds)
    assert crowns.shape == (7, 7)
    assert crowns.dtype == np.int32

    expected = np.zeros((7, 7), dtype=np.int32)
    expected[2:5, 2:5] = 1   # inner 3x3 only
    np.testing.assert_array_equal(crowns, expected)


# ---------------------------------------------------------------------------
# task_plan #3 — max_cr_factor extremes.
# ---------------------------------------------------------------------------


def test_silva_max_cr_factor_large_lets_full_voronoi_cell_through():
    # max_cr_factor=10 → dist threshold = 10*hmax = 100, beats any
    # in-raster distance. Then only exclusion gates. Use a CHM with
    # uniform z=5 except center z=8 → exclusion*hmax = 0.3*8 = 2.4;
    # all cells (z >= 2.4) pass.
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    chm[2, 2] = 8.0
    transform = (0.0, 4.0, 1.0)
    seeds = np.array([[2.0, 2.0, 8.0]], dtype=np.float64)

    crowns = pylidar.segment_silva2016(
        chm, transform, seeds, max_cr_factor=10.0)
    assert int((crowns != 0).sum()) == 25


def test_silva_max_cr_factor_tiny_collapses_to_seed_cell():
    # max_cr_factor=0.01 → dist threshold = 0.01*hmax ≈ tiny. Only
    # the seed cell itself (dist=0) survives.
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    chm[2, 2] = 8.0
    transform = (0.0, 4.0, 1.0)
    seeds = np.array([[2.0, 2.0, 8.0]], dtype=np.float64)

    crowns = pylidar.segment_silva2016(
        chm, transform, seeds, max_cr_factor=0.01)
    expected = np.zeros((5, 5), dtype=np.int32)
    expected[2, 2] = 1
    np.testing.assert_array_equal(crowns, expected)


# ---------------------------------------------------------------------------
# task_plan #4 — single seed Voronoi covers all non-NaN cells.
# ---------------------------------------------------------------------------


def test_silva_single_seed_every_non_nan_cell_in_one_voronoi_group():
    # All cells share one nearest-seed group; hmax = max(z) over all
    # cells. Default exclusion=0.3, max_cr_factor=0.6.
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    chm[2, 2] = 10.0
    # Make one cell NaN so we can verify it stays 0.
    chm[0, 0] = np.nan
    transform = (0.0, 4.0, 1.0)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)

    crowns = pylidar.segment_silva2016(chm, transform, seeds)
    # NaN cell must be 0; every other cell that passes (5 >= 0.3*10 = 3
    # ✓ and dist <= 0.6*10 = 6 ✓ across this 5x5) gets the seed's id.
    assert crowns[0, 0] == 0
    assert int((crowns == 1).sum()) == 24


# ---------------------------------------------------------------------------
# task_plan #5 — exclusion is open (0, 1). Both endpoints raise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_silva_exclusion_outside_open_interval_raises(bad):
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="exclusion"):
        pylidar.segment_silva2016(
            chm, (0.0, 4.0, 1.0), seeds, exclusion=bad)


# ---------------------------------------------------------------------------
# Extras — validators / API edge cases.
# ---------------------------------------------------------------------------


def test_silva_chm_wrong_dtype_raises_type_error():
    chm = np.zeros((5, 5), dtype=np.float32)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    with pytest.raises(TypeError, match="float64"):
        pylidar.segment_silva2016(chm, (0.0, 4.0, 1.0), seeds)


def test_silva_seeds_wrong_dtype_raises_type_error():
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        pylidar.segment_silva2016(chm, (0.0, 4.0, 1.0), seeds)


def test_silva_empty_seeds_returns_zero_raster():
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    seeds = np.empty((0, 3), dtype=np.float64)
    crowns = pylidar.segment_silva2016(chm, (0.0, 4.0, 1.0), seeds)
    assert crowns.shape == (5, 5)
    assert crowns.dtype == np.int32
    np.testing.assert_array_equal(crowns, np.zeros((5, 5), dtype=np.int32))


def test_silva_M3_auto_id_assigns_sequential():
    # Two well-separated peaks — each Voronoi cell covers its half. Auto-ID
    # in the (M,3) form yields IDs 1, 2 in input order.
    chm = np.full((5, 9), 5.0, dtype=np.float64)
    chm[2, 1] = 10.0
    chm[2, 7] = 10.0
    transform = (0.0, 4.0, 1.0)
    seeds = np.array(
        [
            [1.0, 2.0, 10.0],   # left peak → auto-id 1
            [7.0, 2.0, 10.0],   # right peak → auto-id 2
        ],
        dtype=np.float64,
    )
    crowns = pylidar.segment_silva2016(chm, transform, seeds)
    # At default thresholds (exclusion=0.3, max_cr_factor=0.6) the whole
    # 5x9 raster is labelled (z=5 ≥ 0.3*10; max in-raster dist is ≪
    # 0.6*10), so the unique set is {1, 2} not {0, 1, 2}.
    assert set(np.unique(crowns).tolist()) == {1, 2}
    # Left half (cols 0..3) under seed 1; right half (cols 5..8) under seed 2.
    assert crowns[2, 1] == 1
    assert crowns[2, 7] == 2


def test_silva_M4_custom_ids_pass_through():
    chm = np.full((5, 9), 5.0, dtype=np.float64)
    chm[2, 1] = 10.0
    chm[2, 7] = 10.0
    transform = (0.0, 4.0, 1.0)
    seeds = np.array(
        [
            [1.0, 2.0, 10.0, 42.0],
            [7.0, 2.0, 10.0, 99.0],
        ],
        dtype=np.float64,
    )
    crowns = pylidar.segment_silva2016(chm, transform, seeds)
    assert set(np.unique(crowns).tolist()) == {42, 99}
    assert crowns[2, 1] == 42
    assert crowns[2, 7] == 99


def test_silva_chm_with_nan_cells_skipped_not_crashed():
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    chm[2, 2] = 10.0
    chm[0, 0] = np.nan
    chm[4, 4] = np.nan
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    crowns = pylidar.segment_silva2016(chm, (0.0, 4.0, 1.0), seeds)
    assert crowns[0, 0] == 0
    assert crowns[4, 4] == 0


def test_silva_seed_outside_chm_silently_dropped():
    # First seed off the raster; second seed in-bounds. The first must
    # not affect anything (it's filtered to the chm bbox before KDTree
    # construction). Note: this is a (M,4) custom-id case so it dodges
    # the Phase 3 D2 (M,3 auto-id) issue — see the xfail below for D2.
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    chm[2, 2] = 10.0
    seeds = np.array(
        [
            [9999.0, 9999.0, 5.0, 7.0],
            [2.0,    2.0,    10.0, 99.0],
        ],
        dtype=np.float64,
    )
    crowns = pylidar.segment_silva2016(chm, (0.0, 4.0, 1.0), seeds)
    # Surviving seed has id=99; id=7 must not appear.
    labels = set(np.unique(crowns).tolist())
    assert 7 not in labels
    assert 99 in labels


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                 0.0, -1.0])
def test_silva_max_cr_factor_invalid_scalars_raise(bad):
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="max_cr_factor"):
        pylidar.segment_silva2016(
            chm, (0.0, 4.0, 1.0), seeds, max_cr_factor=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_silva_exclusion_nonfinite_raises(bad):
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="exclusion"):
        pylidar.segment_silva2016(
            chm, (0.0, 4.0, 1.0), seeds, exclusion=bad)


def test_silva_default_output_dtype_is_int32():
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    crowns = pylidar.segment_silva2016(chm, (0.0, 4.0, 1.0), seeds)
    assert crowns.dtype == np.int32


# ---------------------------------------------------------------------------
# C++ self-check — bypass the Python validator and confirm the core
# raises on its own. Spec §6.4: core/ must validate scalar invariants
# even when linked outside the Python wrapper. Same template as
# Phase 1/2 (smooth_height / lmf hmin).
# ---------------------------------------------------------------------------


def test_core_silva2016_rejects_nonfinite_max_cr_factor_directly():
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.silva2016(chm, 0.0, 4.0, 1.0, seeds,
                        float("nan"), 0.3)


def test_core_silva2016_rejects_exclusion_at_one_directly():
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.silva2016(chm, 0.0, 4.0, 1.0, seeds,
                        0.6, 1.0)


@pytest.mark.parametrize(
    "bad_x,bad_y",
    [
        (float("nan"), 2.0),
        (2.0, float("nan")),
        (float("inf"), 2.0),
        (2.0, float("-inf")),
    ],
)
def test_core_silva2016_rejects_nonfinite_seed_xy_directly(bad_x, bad_y):
    """Spec §6.4 / silva2016.hpp contract: when called directly via _core
    (bypassing the Python validator), silva2016 must throw on non-finite
    seed XY rather than silently drop. Silent drops would yield
    mysterious empty / partial results when core/ is linked outside the
    Python wrapper."""
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    seeds = np.array([[bad_x, bad_y, 10.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        _core.silva2016(chm, 0.0, 4.0, 1.0, seeds, 0.6, 0.3)


# ---------------------------------------------------------------------------
# Phase 3 D2 deferred fix — pin the divergence so it gets unxfailed when
# Phase 8 ships the auto-ID fix.
#
# lidR semantics (R/algorithm-its.R:240-243 then 260):
#   res <- crop_special_its(treetops, chm, bbox)        # crop FIRST
#   ids <- treetops[[ID]]                                # then read ids
# meaning if a user passes seeds without an ID column, lidR's
# check_tree_tops would assign 1:n AFTER the crop. pylidar currently
# (Phase 3-4) auto-IDs in Python BEFORE the C++ side drops out-of-bbox
# seeds, so the surviving crown carries the seed's *original* index, not
# its post-crop sequential index.
#
# strict=True: when D2 is fixed in Phase 8 the assertion will pass and
# pytest will fail the run (XPASS), prompting removal of the xfail
# marker. See findings.md "Phase 3 dalponte2016 已确认偏离 / Deferred
# fixes" item D2 for the full discussion.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=("Phase 3 deferred fix D2: (M,3) auto-ID is assigned in "
            "Python before C++ drops out-of-bbox seeds, so the crown "
            "label is 2 instead of the lidR-correct 1. To be fixed in "
            "Phase 8 with the fixture-comparison suite."),
    strict=True,
)
def test_silva_M3_first_seed_outside_bbox_crown_label_matches_lidR():
    chm = np.full((5, 5), 5.0, dtype=np.float64)
    chm[2, 2] = 10.0
    seeds = np.array(
        [
            [9999.0, 9999.0, 5.0],   # out-of-bbox: lidR would crop it out
            [2.0,    2.0,    10.0],  # in-bounds: lidR assigns id=1
        ],
        dtype=np.float64,
    )
    crowns = pylidar.segment_silva2016(chm, (0.0, 4.0, 1.0), seeds)
    # lidR-correct: surviving seed gets id=1 (because it's the only
    # post-crop seed). pylidar today: id=2 (auto-IDed before crop).
    labels = sorted(np.unique(crowns[crowns > 0]).tolist())
    assert labels == [1]
