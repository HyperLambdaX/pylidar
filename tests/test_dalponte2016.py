"""Phase 3 acceptance tests — segment_dalponte2016.

Five task_plan-required cases:
  1. Single peak / single seed → labelled crown of expected shape.
  2. Two peaks / two seeds → two distinct, non-overlapping crowns.
  3. Seed in an above-th_tree mask: a th_tree-blocked direction stops
     crown growth on that side. Verifies the th_tree mask actually
     gates expansion.
  4. Custom IDs supplied via (M, 4) seeds → output uses caller IDs.
  5. Wrong dtype (float32 chm) → TypeError.

Plus extras for the validators (seeds dtype, M=0 short-circuit,
out-of-CHM seed dropped, NaN cells survive, th_seed out of [0,1],
seed id == 0 rejected, default output dtype).
"""

from __future__ import annotations

import numpy as np
import pytest

import pylidar


# ---------------------------------------------------------------------------
# Hand-traced expected outputs.
#
# The algorithm scans each labelled pixel in row-major order, looks at its 4
# cardinal neighbours, and adds one if it passes:
#   z > th_tree
#   z > h_seed * th_seed
#   z > mean_crown_z * th_cr
#   z <= h_seed * 1.05
#   |seed.row - nr| < max_cr  &&  |seed.col - nc| < max_cr
#   region[nr, nc] == 0
# The outer loop runs until no scan changes any label. Within a single scan
# multiple seeds can both write into the same neighbour cell — the
# row-major-last writer wins. Border (row 0/H-1, col 0/W-1) pixels are never
# *sources* of expansion (matches lidR's `r in [1, nrow-2]` loop bounds).
# ---------------------------------------------------------------------------


def test_single_peak_single_seed_grows_3x3_crown():
    chm = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 6, 6, 6, 0],
            [0, 6, 10, 6, 0],
            [0, 6, 6, 6, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    transform = (0.0, 4.0, 1.0)
    # World XY of pixel (row=2, col=2): (2.0, 2.0).
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)

    crowns = pylidar.segment_dalponte2016(
        chm, transform=transform, seeds=seeds
    )
    assert crowns.shape == (5, 5)
    assert crowns.dtype == np.int32

    expected = np.zeros((5, 5), dtype=np.int32)
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(crowns, expected)


def test_two_peaks_two_seeds_yield_distinct_crowns():
    # Ring height 7 (not 6) so the right peak (h_seed=12) also passes
    # th_cr: 7 > 12 * 0.55 = 6.6. Left peak (h_seed=10) passes either way
    # (7 > 10 * 0.55 = 5.5).
    chm = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 7, 7, 7, 0, 7, 7, 7, 0],
            [0, 7, 10, 7, 0, 7, 12, 7, 0],
            [0, 7, 7, 7, 0, 7, 7, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    transform = (0.0, 4.0, 1.0)
    seeds = np.array(
        [
            [2.0, 2.0, 10.0],   # → row 2, col 2 → id 1
            [6.0, 2.0, 12.0],   # → row 2, col 6 → id 2
        ],
        dtype=np.float64,
    )

    crowns = pylidar.segment_dalponte2016(
        chm, transform=transform, seeds=seeds
    )
    expected = np.zeros((5, 9), dtype=np.int32)
    expected[1:4, 1:4] = 1
    expected[1:4, 5:8] = 2
    np.testing.assert_array_equal(crowns, expected)


def test_th_tree_mask_blocks_growth_in_one_direction():
    # Row 3 sits at z=1 < th_tree=2 → crown can't extend down. Rows 1-2 are
    # all >= th_tree and form a clean rectangular crown.
    chm = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 6, 6, 6, 0],
            [0, 6, 10, 6, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    transform = (0.0, 4.0, 1.0)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)

    crowns = pylidar.segment_dalponte2016(
        chm, transform=transform, seeds=seeds, th_tree=2.0
    )
    expected = np.zeros((5, 5), dtype=np.int32)
    expected[1:3, 1:4] = 1   # 2 rows × 3 cols only
    np.testing.assert_array_equal(crowns, expected)


def test_custom_ids_pass_through_to_output_raster():
    chm = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 7, 7, 7, 0, 7, 7, 7, 0],
            [0, 7, 10, 7, 0, 7, 12, 7, 0],
            [0, 7, 7, 7, 0, 7, 7, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    transform = (0.0, 4.0, 1.0)
    # (M, 4) form: caller-provided IDs (last column).
    seeds = np.array(
        [
            [2.0, 2.0, 10.0, 42.0],
            [6.0, 2.0, 12.0, 99.0],
        ],
        dtype=np.float64,
    )

    crowns = pylidar.segment_dalponte2016(
        chm, transform=transform, seeds=seeds
    )
    expected = np.zeros((5, 9), dtype=np.int32)
    expected[1:4, 1:4] = 42
    expected[1:4, 5:8] = 99
    np.testing.assert_array_equal(crowns, expected)


def test_chm_wrong_dtype_raises_type_error():
    chm = np.zeros((5, 5), dtype=np.float32)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    with pytest.raises(TypeError, match="float64"):
        pylidar.segment_dalponte2016(
            chm, transform=(0.0, 4.0, 1.0), seeds=seeds
        )


# ---------------------------------------------------------------------------
# Extras — validators / API edge cases.
# ---------------------------------------------------------------------------


def test_seeds_wrong_dtype_raises_type_error():
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        pylidar.segment_dalponte2016(
            chm, transform=(0.0, 4.0, 1.0), seeds=seeds
        )


def test_seeds_wrong_shape_raises_value_error():
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0]], dtype=np.float64)  # (M, 2)
    with pytest.raises(ValueError, match=r"\(M, 3\) or \(M, 4\)"):
        pylidar.segment_dalponte2016(
            chm, transform=(0.0, 4.0, 1.0), seeds=seeds
        )


def test_empty_seeds_returns_zero_raster():
    chm = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 6, 6, 6, 0],
            [0, 6, 10, 6, 0],
            [0, 6, 6, 6, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    seeds = np.empty((0, 3), dtype=np.float64)
    crowns = pylidar.segment_dalponte2016(
        chm, transform=(0.0, 4.0, 1.0), seeds=seeds
    )
    assert crowns.shape == (5, 5)
    assert crowns.dtype == np.int32
    np.testing.assert_array_equal(crowns, np.zeros((5, 5), dtype=np.int32))


def test_seed_outside_chm_is_silently_skipped():
    # One in-bounds seed (grows a crown) + one well off the raster
    # (silently dropped, matching lidR's crop_special_its semantics).
    chm = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 6, 6, 6, 0],
            [0, 6, 10, 6, 0],
            [0, 6, 6, 6, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    seeds = np.array(
        [
            [2.0,    2.0, 10.0],
            [9999.0, 9999.0, 5.0],
        ],
        dtype=np.float64,
    )
    crowns = pylidar.segment_dalponte2016(
        chm, transform=(0.0, 4.0, 1.0), seeds=seeds
    )
    expected = np.zeros((5, 5), dtype=np.int32)
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(crowns, expected)


def test_chm_with_nan_cells_does_not_crash():
    # NaN cells are masked (treated as -inf internally). Crown must still
    # grow over the non-NaN, above-th_tree area.
    chm = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 6,      6,      6,      np.nan],
            [np.nan, 6,      10,     6,      np.nan],
            [np.nan, 6,      6,      6,      np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    crowns = pylidar.segment_dalponte2016(
        chm, transform=(0.0, 4.0, 1.0), seeds=seeds
    )
    expected = np.zeros((5, 5), dtype=np.int32)
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(crowns, expected)


@pytest.mark.parametrize(
    "param,value",
    [
        ("th_seed", -0.1),
        ("th_seed", 1.1),
        ("th_cr",  -0.1),
        ("th_cr",   1.5),
    ],
)
def test_th_seed_or_th_cr_out_of_range_raises_value_error(param, value):
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match=param):
        pylidar.segment_dalponte2016(
            chm, transform=(0.0, 4.0, 1.0), seeds=seeds, **{param: value}
        )


def test_max_cr_non_positive_raises_value_error():
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="max_cr"):
        pylidar.segment_dalponte2016(
            chm, transform=(0.0, 4.0, 1.0), seeds=seeds, max_cr=0.0
        )


@pytest.mark.parametrize("bad_id", [0.0, -1.0, -42.0])
def test_seed_id_below_one_in_M4_form_raises_value_error(bad_id):
    """Spec §3 / §7: tree IDs are ≥ 1. id=0 is the "no tree" sentinel and
    negative ids would silently break downstream `crowns > 0` masks."""
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([[2.0, 2.0, 10.0, bad_id]], dtype=np.float64)
    with pytest.raises(ValueError, match=">= 1"):
        pylidar.segment_dalponte2016(
            chm, transform=(0.0, 4.0, 1.0), seeds=seeds
        )


@pytest.mark.parametrize(
    "bad_xyz",
    [
        [float("nan"), 2.0, 10.0],
        [2.0, float("nan"), 10.0],
        [2.0, 2.0, float("nan")],
        [float("inf"), 2.0, 10.0],
        [2.0, float("-inf"), 10.0],
    ],
)
def test_seed_nonfinite_xyz_raises_value_error(bad_xyz):
    """lidR errors at sf::st_rasterize on NaN/Inf point coords; pylidar
    used to silently drop them in C++. Now rejected up-front to match."""
    chm = np.zeros((5, 5), dtype=np.float64)
    seeds = np.array([bad_xyz], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        pylidar.segment_dalponte2016(
            chm, transform=(0.0, 4.0, 1.0), seeds=seeds
        )


def test_max_cr_compactness_constraint_clips_crown():
    # max_cr=2 → Chebyshev distance < 2 means {0, 1}. From the seed at
    # (row=2, col=2), pixels at row/col distance >= 2 are rejected. So in
    # a 5x5 single-peak CHM the crown should shrink to the inner 3x3
    # (just {0,1} steps) — same as the default test, but verify shrinkage
    # against a 7x7 wide-radius peak which would otherwise grow further.
    chm = np.full((7, 7), 6.0, dtype=np.float64)
    chm[3, 3] = 10.0
    # Border ring at z=0 → won't be grown into (fails th_tree=2).
    chm[0, :] = 0.0
    chm[-1, :] = 0.0
    chm[:, 0] = 0.0
    chm[:, -1] = 0.0
    seeds = np.array([[3.0, 3.0, 10.0]], dtype=np.float64)
    transform = (0.0, 6.0, 1.0)  # → (col=3, row=3) for world (3, 3)

    crowns_wide = pylidar.segment_dalponte2016(
        chm, transform=transform, seeds=seeds, max_cr=10.0
    )
    crowns_narrow = pylidar.segment_dalponte2016(
        chm, transform=transform, seeds=seeds, max_cr=2.0
    )
    # Narrow run must label *strictly fewer* pixels than the wide run.
    assert (crowns_narrow != 0).sum() < (crowns_wide != 0).sum()
    # And the seed itself is always labelled.
    assert crowns_narrow[3, 3] == 1


def test_default_output_dtype_is_int32():
    chm = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 6, 6, 6, 0],
            [0, 6, 10, 6, 0],
            [0, 6, 6, 6, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    seeds = np.array([[2.0, 2.0, 10.0]], dtype=np.float64)
    crowns = pylidar.segment_dalponte2016(
        chm, transform=(0.0, 4.0, 1.0), seeds=seeds
    )
    assert crowns.dtype == np.int32
