"""point_mask — boolean mask from lidR-named-macro predicates.

PORT NOTE
---------
Mirrors ``filters.R::filter_poi`` semantics (AND-combine, NA→False) and the
``filter_*`` named-macro vocabulary. The synthetic LAS from ``conftest.py``
gives 12 hand-traceable points: alternating Classification {2,7,5}, two
returns per point with return_number alternating {1,2}, intensities 10..120.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar import io


# ---------------------------------------------------------------- happy

def test_default_keeps_everything(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las)
    assert mask.shape == (12,)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_keep_first(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, keep_first=True)
    assert mask.sum() == 6
    assert np.all(np.asarray(las.return_number)[mask] == 1)


def test_keep_last(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, keep_last=True)
    rn = np.asarray(las.return_number)
    nr = np.asarray(las.number_of_returns)
    assert mask.sum() == 6
    assert np.all(rn[mask] == nr[mask])


def test_keep_single_zero_when_all_have_two_returns(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, keep_single=True)
    assert mask.sum() == 0


def test_keep_single_when_overridden(build_synthetic_las):
    las = build_synthetic_las(
        number_of_returns=np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2], dtype=np.uint8),
        return_numbers=np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1], dtype=np.uint8),
    )
    mask = io.point_mask(las, keep_single=True)
    assert mask.sum() == 6


def test_keep_firstofmany(build_synthetic_las):
    las = build_synthetic_las()
    # Default has number_of_returns == 2 everywhere, return_number alternating
    # 1/2 → first-of-many fires for the 6 first-return rows.
    mask = io.point_mask(las, keep_firstofmany=True)
    assert mask.sum() == 6


def test_keep_firstlast(build_synthetic_las):
    las = build_synthetic_las()
    # All points are either ReturnNumber==1 (first) or ==NumberOfReturns==2 (last)
    mask = io.point_mask(las, keep_firstlast=True)
    assert mask.all()


def test_keep_ground(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, keep_ground=True)
    cls = np.asarray(las.classification)
    assert mask.sum() == 4
    assert np.all(cls[mask] == 2)


def test_keep_class_scalar(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, keep_class=5)
    assert mask.sum() == 4
    assert np.all(np.asarray(las.classification)[mask] == 5)


def test_keep_class_sequence(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, keep_class=[2, 5])
    cls = np.asarray(las.classification)
    assert mask.sum() == 8
    assert set(np.unique(cls[mask]).tolist()) == {2, 5}


def test_drop_class_scalar(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, drop_class=7)
    cls = np.asarray(las.classification)
    assert mask.sum() == 8
    assert 7 not in cls[mask]


def test_drop_class_sequence(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, drop_class=[5, 7])
    assert mask.sum() == 4
    assert np.all(np.asarray(las.classification)[mask] == 2)


def test_drop_z_below(build_synthetic_las):
    las = build_synthetic_las()
    # z = 0, 0.5, 1, 1.5, 2, 2.5, ... drop below 2.0 keeps 8 points (z >= 2.0)
    mask = io.point_mask(las, drop_z_below=2.0)
    z = np.asarray(las.z)
    assert mask.sum() == 8
    assert np.all(z[mask] >= 2.0)


def test_drop_z_above(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, drop_z_above=2.0)
    z = np.asarray(las.z)
    assert mask.sum() == 5
    assert np.all(z[mask] <= 2.0)


def test_drop_intensity_below(build_synthetic_las):
    las = build_synthetic_las()
    # intensity = 10, 20, ..., 120; below 50 → keep 8 (>= 50)
    mask = io.point_mask(las, drop_intensity_below=50)
    assert mask.sum() == 8
    assert np.all(np.asarray(las.intensity)[mask] >= 50)


def test_drop_intensity_above(build_synthetic_las):
    las = build_synthetic_las()
    mask = io.point_mask(las, drop_intensity_above=50)
    assert mask.sum() == 5


def test_keep_xy(build_synthetic_las):
    las = build_synthetic_las()
    # x ∈ {0, 1, 2, 3}, y ∈ {0, 1, 2}; keep x ∈ [1, 2], y ∈ [0, 1] → 4 points
    mask = io.point_mask(las, keep_xy=(1.0, 0.0, 2.0, 1.0))
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    assert mask.sum() == 4
    assert np.all((x[mask] >= 1) & (x[mask] <= 2))
    assert np.all((y[mask] >= 0) & (y[mask] <= 1))


# ---------------------------------------------------------------- AND-combine

def test_kwargs_and_combine(build_synthetic_las):
    las = build_synthetic_las()
    # keep_first (6 pts: indices 0,2,4,6,8,10) AND drop_class=[7] (drops 7s only)
    # First-return classifications by index: 0→2, 2→5, 4→7, 6→2, 8→5, 10→7
    # So keep_first AND drop_class=[7] keeps indices 0, 2, 6, 8 → 4 points
    mask = io.point_mask(las, keep_first=True, drop_class=[7])
    assert mask.sum() == 4
    assert np.all(np.asarray(las.return_number)[mask] == 1)
    assert 7 not in np.asarray(las.classification)[mask]


def test_kwargs_three_way_and(build_synthetic_las):
    las = build_synthetic_las()
    # keep_first (z = 0, 1, 2, 3, 4, 5) AND drop_z_below=2.0 (z >= 2 → 4 pts)
    # AND drop_class=[7] → first-return ground-or-veg only:
    #   first-return indices 4 (z=2, class 7) drops, 6 (z=3, class 2) keeps,
    #   8 (z=4, class 5) keeps, 10 (z=5, class 7) drops → 2 points
    mask = io.point_mask(las, keep_first=True, drop_z_below=2.0, drop_class=[7])
    assert mask.sum() == 2


# ---------------------------------------------------------------- error gating

def test_unknown_kwarg_raises(build_synthetic_las):
    las = build_synthetic_las()
    with pytest.raises(io.FilterKwargError):
        io.point_mask(las, totally_invalid_filter=True)


def test_keep_xy_wrong_length_raises(build_synthetic_las):
    las = build_synthetic_las()
    with pytest.raises(ValueError):
        io.point_mask(las, keep_xy=(0.0, 0.0, 1.0))


def test_keep_class_invalid_type_raises(build_synthetic_las):
    las = build_synthetic_las()
    with pytest.raises(TypeError):
        io.point_mask(las, keep_class="not-an-int")
