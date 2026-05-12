"""read_las — laspy round-trip with select DSL and filter kwargs.

PORT NOTE
---------
Verifies the lidR ``readLAS`` semantic mapping documented in
``python/pylidar/io.py``. The synthetic LAS produced by ``conftest.py``
is the smallest fixture that exercises every kwarg branch deterministically
(12 points, alternating ground/noise/veg, two returns per point).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar import io


# ---------------------------------------------------------------- happy path

def test_read_las_basic(synthetic_las_path):
    las = io.read_las(synthetic_las_path)
    assert len(las.x) == 12
    assert hasattr(las, "user_select")


def test_read_las_keep_first_drops_seconds(synthetic_las_path):
    las = io.read_las(synthetic_las_path, keep_first=True)
    assert len(las.x) == 6
    assert np.all(np.asarray(las.return_number) == 1)


def test_read_las_drop_class_noise(synthetic_las_path):
    las = io.read_las(synthetic_las_path, drop_class=[7])
    assert len(las.x) == 8
    assert 7 not in np.asarray(las.classification)


def test_read_las_keep_first_and_drop_z_below(synthetic_las_path):
    # First returns are at indices 0, 2, 4, 6, 8, 10 → z = 0, 1, 2, 3, 4, 5
    # drop_z_below=2.5 keeps z ∈ {3, 4, 5} → 3 points
    las = io.read_las(synthetic_las_path, keep_first=True, drop_z_below=2.5)
    assert len(las.x) == 3
    z = np.asarray(las.z)
    assert np.all(z >= 2.5)
    assert np.all(np.asarray(las.return_number) == 1)


def test_read_las_keep_xy_bbox(synthetic_las_path):
    # xy lattice is 4×3 with x ∈ {0,1,2,3}, y ∈ {0,1,2}; bbox keeps x∈[1,2], y∈[0,1]
    las = io.read_las(synthetic_las_path, keep_xy=(1.0, 0.0, 2.0, 1.0))
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    assert len(x) == 4
    assert np.all((x >= 1.0) & (x <= 2.0))
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_read_las_keep_class_multiple(synthetic_las_path):
    # ground + veg only (drop noise)
    las = io.read_las(synthetic_las_path, keep_class=[2, 5])
    cls = np.asarray(las.classification)
    assert set(np.unique(cls).tolist()).issubset({2, 5})
    assert len(cls) == 8


# ---------------------------------------------------------------- select DSL

def test_select_default_is_wildcard():
    fields, _, all_extras, dropped = io.parse_select(None)
    assert {"X", "Y", "Z", "intensity", "return_number"} <= fields
    assert all_extras
    assert dropped == frozenset()


def test_select_wildcard_string():
    fields, _, all_extras, dropped = io.parse_select("*")
    assert "intensity" in fields
    assert all_extras
    assert dropped == frozenset()


def test_select_minimal_xyz():
    fields, _, all_extras, dropped = io.parse_select("xyz")
    assert fields == frozenset({"X", "Y", "Z"})
    assert not all_extras
    assert dropped == frozenset()


def test_select_letter_dsl_maps_chars():
    fields, _, _, _ = io.parse_select("xyziar")
    assert fields == frozenset({"X", "Y", "Z", "intensity", "scan_angle", "return_number"})


def test_select_negation():
    fields, _, _, _ = io.parse_select("* -i -a")
    assert "intensity" not in fields
    assert "scan_angle" not in fields
    assert "return_number" in fields  # still present
    assert {"X", "Y", "Z"} <= fields  # mandatory


def test_select_negation_cannot_drop_xyz():
    fields, _, _, _ = io.parse_select("* -x -y -z")
    assert {"X", "Y", "Z"} <= fields  # mandatory survive negation


def test_select_extra_bytes_all():
    fields, extras, all_extras, dropped = io.parse_select("xyz0")
    assert all_extras
    assert extras == frozenset()
    assert dropped == frozenset()


def test_select_extra_bytes_ordinals():
    fields, extras, all_extras, dropped = io.parse_select("xyz123")
    assert extras == frozenset({1, 2, 3})
    assert not all_extras
    assert dropped == frozenset()


def test_select_negate_extra_byte_under_wildcard():
    """`* -1` — keep all extras EXCEPT first. Reviewer's bug:
    previously `all_extras` masked this case and `extras_dropped` was lost.
    """
    _, extras, all_extras, dropped = io.parse_select("* -1")
    assert all_extras  # still wildcard
    assert extras == frozenset()  # no explicit kept
    assert dropped == frozenset({1})  # but #1 is dropped


def test_select_negate_zero_clears_wildcard():
    """`-0` after `*` resets to "no extras" — neither all nor any specific."""
    _, extras, all_extras, dropped = io.parse_select("* -0")
    assert not all_extras
    assert extras == frozenset()
    assert dropped == frozenset()


def test_select_full_waveform_raises():
    with pytest.raises(NotImplementedError):
        io.parse_select("xyzW")


def test_select_unknown_char_raises():
    with pytest.raises(ValueError):
        io.parse_select("xyz?")


def test_select_sequence_input():
    fields, _, _, _ = io.parse_select(["X", "Y", "Z", "intensity"])
    assert fields == frozenset({"X", "Y", "Z", "intensity"})


def test_select_sequence_normalizes_aliases():
    fields, _, _, _ = io.parse_select(["x", "y", "z", "scan_angle_rank"])
    assert "scan_angle" in fields  # alias collapses


def test_select_sequence_unknown_raises():
    with pytest.raises(ValueError):
        io.parse_select(["X", "Y", "Z", "totally_made_up"])


# ---------------------------------------------------------------- error gating

def test_read_las_laslib_filter_stub_raises(synthetic_las_path):
    with pytest.raises(NotImplementedError):
        io.read_las(synthetic_las_path, laslib_filter="-keep_first")


def test_read_las_unknown_kwarg_raises(synthetic_las_path):
    with pytest.raises(io.FilterKwargError):
        io.read_las(synthetic_las_path, keep_only_pretty_points=True)


def test_read_las_records_user_select(synthetic_las_path):
    las = io.read_las(synthetic_las_path, select="xyzi")
    assert las.user_select == frozenset({"X", "Y", "Z", "intensity"})


def test_read_las_empty_select_treated_as_xyz():
    fields, _, _, _ = io.parse_select("")
    assert fields == frozenset({"X", "Y", "Z"})
