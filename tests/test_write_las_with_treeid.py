"""Tests for ``pylidar.io.write_las_with_treeid``."""

from __future__ import annotations

import numpy as np
import laspy
import pytest

import pylidar
from laspy.vlrs.known import GeoKeyDirectoryVlr


def _las_with_geokeys(build_synthetic_las):
    """Build a synthetic LAS and stuff a GeoKeyDirectoryVlr in so we can
    confirm CRS-style VLRs survive the round-trip."""
    las = build_synthetic_las()
    las.header.vlrs.append(GeoKeyDirectoryVlr())
    return las


def test_writer_round_trip_treeid_int32(build_synthetic_las, tmp_path):
    las = _las_with_geokeys(build_synthetic_las)
    n = len(las.x)
    tree_id = np.arange(n, dtype=np.int32)
    tree_id[0] = np.iinfo(np.int32).max  # NA sentinel
    out_path = tmp_path / "out.las"

    pylidar.io.write_las_with_treeid(las, tree_id, out_path)

    rd = laspy.read(str(out_path))
    assert "treeID" in rd.point_format.dimension_names
    assert rd.treeID.dtype == np.int32
    eb = rd.point_format.dimension_by_name("treeID")
    assert eb.num_bytes == 4
    assert list(np.asarray(rd.treeID)) == list(tree_id)


def test_writer_preserves_vlr_and_crs_metadata(build_synthetic_las, tmp_path):
    las = _las_with_geokeys(build_synthetic_las)
    n = len(las.x)
    out_path = tmp_path / "out.las"

    pylidar.io.write_las_with_treeid(
        las, np.zeros(n, dtype=np.int32), out_path
    )
    rd = laspy.read(str(out_path))

    geokey_vlrs = [v for v in rd.header.vlrs if isinstance(v, GeoKeyDirectoryVlr)]
    assert geokey_vlrs, "GeoKeyDirectoryVlr lost during write"


def test_writer_preserves_existing_point_dimensions(build_synthetic_las, tmp_path):
    las = build_synthetic_las()
    n = len(las.x)
    out_path = tmp_path / "out.las"
    pylidar.io.write_las_with_treeid(
        las, np.zeros(n, dtype=np.int32), out_path
    )
    rd = laspy.read(str(out_path))
    assert list(np.asarray(rd.intensity)) == list(np.asarray(las.intensity))
    assert list(np.asarray(rd.classification)) == list(np.asarray(las.classification))
    assert list(np.asarray(rd.return_number)) == list(np.asarray(las.return_number))


def test_writer_preserves_header_scales_offsets_and_format(
    build_synthetic_las, tmp_path
):
    las = build_synthetic_las()
    las.header.scales = np.array([0.01, 0.02, 0.03])
    las.header.offsets = np.array([1.5, 2.5, 3.5])
    n = len(las.x)
    out_path = tmp_path / "out.las"
    pylidar.io.write_las_with_treeid(
        las, np.zeros(n, dtype=np.int32), out_path
    )
    rd = laspy.read(str(out_path))
    assert rd.point_format.id == las.point_format.id
    assert tuple(rd.header.scales) == (0.01, 0.02, 0.03)
    assert tuple(rd.header.offsets) == (1.5, 2.5, 3.5)


def test_writer_na_sentinel_is_int32_max(build_synthetic_las, tmp_path):
    las = build_synthetic_las()
    n = len(las.x)
    out_path = tmp_path / "out.las"
    pylidar.io.write_las_with_treeid(
        las, np.zeros(n, dtype=np.int32), out_path
    )
    rd = laspy.read(str(out_path))
    # The ExtraBytesVlr carries the no_data sentinel.
    eb_vlrs = [
        v for v in rd.header.vlrs if v.__class__.__name__ == "ExtraBytesVlr"
    ]
    assert eb_vlrs
    eb = eb_vlrs[0].extra_bytes_structs[0]
    assert int(eb.no_data[0]) == np.iinfo(np.int32).max


def test_writer_float64_path_uses_double_extra_dim(
    build_synthetic_las, tmp_path
):
    las = build_synthetic_las()
    n = len(las.x)
    tree_id = np.array([float(i) for i in range(n)], dtype=np.float64)
    out_path = tmp_path / "out.las"

    pylidar.io.write_las_with_treeid(
        las, tree_id, out_path, uniqueness="bitmerge"
    )
    rd = laspy.read(str(out_path))
    assert rd.treeID.dtype == np.float64
    eb = rd.point_format.dimension_by_name("treeID")
    assert eb.num_bytes == 8
    assert list(np.asarray(rd.treeID)) == list(tree_id)


def test_writer_float64_na_sentinel_is_double_xmin(
    build_synthetic_las, tmp_path
):
    las = build_synthetic_las()
    n = len(las.x)
    tree_id = np.zeros(n, dtype=np.float64)
    out_path = tmp_path / "out.las"
    pylidar.io.write_las_with_treeid(
        las, tree_id, out_path, uniqueness="gpstime"
    )
    rd = laspy.read(str(out_path))
    eb_vlrs = [
        v for v in rd.header.vlrs if v.__class__.__name__ == "ExtraBytesVlr"
    ]
    eb = eb_vlrs[0].extra_bytes_structs[0]
    # R's .Machine$double.xmin ≈ np.finfo(float64).tiny
    assert float(eb.no_data[0]) == pytest.approx(np.finfo(np.float64).tiny)


def test_writer_optional_rgb_persisted(build_synthetic_las, tmp_path):
    las = build_synthetic_las()
    n = len(las.x)
    rgb = np.column_stack(
        [
            np.arange(n, dtype=np.uint16) * 100,
            np.arange(n, dtype=np.uint16) * 200,
            np.arange(n, dtype=np.uint16) * 300,
        ]
    )
    out_path = tmp_path / "out.las"
    pylidar.io.write_las_with_treeid(
        las, np.zeros(n, dtype=np.int32), out_path, rgb=rgb
    )
    rd = laspy.read(str(out_path))
    assert list(np.asarray(rd.red)) == list(rgb[:, 0])
    assert list(np.asarray(rd.green)) == list(rgb[:, 1])
    assert list(np.asarray(rd.blue)) == list(rgb[:, 2])


def test_writer_rgb_wrong_shape_raises(build_synthetic_las, tmp_path):
    las = build_synthetic_las()
    n = len(las.x)
    bad_rgb = np.zeros((n, 2), dtype=np.uint16)
    out_path = tmp_path / "out.las"
    with pytest.raises(ValueError, match=r"\(.*3\)"):
        pylidar.io.write_las_with_treeid(
            las, np.zeros(n, dtype=np.int32), out_path, rgb=bad_rgb
        )


def test_writer_length_mismatch_raises(build_synthetic_las, tmp_path):
    las = build_synthetic_las()
    n = len(las.x)
    out_path = tmp_path / "out.las"
    with pytest.raises(ValueError, match="point count"):
        pylidar.io.write_las_with_treeid(
            las, np.zeros(n - 1, dtype=np.int32), out_path
        )


def test_writer_bad_dtype_raises(build_synthetic_las, tmp_path):
    las = build_synthetic_las()
    n = len(las.x)
    out_path = tmp_path / "out.las"
    with pytest.raises(TypeError, match="int32 or float64"):
        pylidar.io.write_las_with_treeid(
            las, np.zeros(n, dtype=np.int16), out_path
        )


def test_writer_unknown_uniqueness_raises(build_synthetic_las, tmp_path):
    las = build_synthetic_las()
    n = len(las.x)
    out_path = tmp_path / "out.las"
    with pytest.raises(ValueError, match="uniqueness"):
        pylidar.io.write_las_with_treeid(
            las, np.zeros(n, dtype=np.int32), out_path, uniqueness="bogus"
        )


def test_writer_rejects_rgb_on_non_rgb_format(build_synthetic_las, tmp_path):
    las = build_synthetic_las(point_format=1)  # format 1: gpstime, no RGB
    n = len(las.x)
    out_path = tmp_path / "out.las"
    rgb = np.zeros((n, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match="point_format"):
        pylidar.io.write_las_with_treeid(
            las, np.zeros(n, dtype=np.int32), out_path, rgb=rgb
        )


def test_writer_idempotent_on_input_las(build_synthetic_las, tmp_path):
    """Writing must not mutate the input LasData."""
    las = build_synthetic_las()
    pre_dims = set(las.point_format.dimension_names)
    pre_intensity = np.asarray(las.intensity).copy()
    n = len(las.x)
    pylidar.io.write_las_with_treeid(
        las, np.zeros(n, dtype=np.int32), tmp_path / "out.las"
    )
    assert set(las.point_format.dimension_names) == pre_dims
    assert "treeID" not in las.point_format.dimension_names
    assert list(np.asarray(las.intensity)) == list(pre_intensity)


def test_writer_rejects_when_treeid_already_present(
    build_synthetic_las, tmp_path
):
    las = build_synthetic_las()
    las.add_extra_dim(
        laspy.ExtraBytesParams(name="treeID", type=np.int32)
    )
    n = len(las.x)
    with pytest.raises(ValueError, match="treeID"):
        pylidar.io.write_las_with_treeid(
            las, np.zeros(n, dtype=np.int32), tmp_path / "out.las"
        )


# ---------------------------------------------------------------- Phase 5 audit-fix #1
# Apex-based gpstime / bitmerge recomputation per lidR segment_trees.R:68-108.


def _las_with_gpstime(build_synthetic_las, gpstime_values=None):
    """Build an LAS where gpstime can be set deterministically."""
    las = build_synthetic_las()
    n = len(las.x)
    if gpstime_values is None:
        gpstime_values = np.arange(n, dtype=np.float64) + 100.0
    las.gps_time = gpstime_values
    return las


def test_writer_gpstime_recomputes_apex_id(build_synthetic_las, tmp_path):
    """Per-tree apex (max-z) gpstime becomes the persistent ID."""
    las = _las_with_gpstime(build_synthetic_las)
    n = len(las.x)
    # Two trees: 6 points each. The apex (max z within group) drives the ID.
    tree_id = np.zeros(n, dtype=np.int32)
    tree_id[:6] = 1
    tree_id[6:] = 2

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="gpstime"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    assert rd.treeID.dtype == np.float64

    z = np.asarray(las.z)
    t = np.asarray(las.gps_time)
    expected_g1 = t[np.argmax(z[:6])]
    expected_g2 = t[6 + np.argmax(z[6:])]
    assert all(np.asarray(rd.treeID)[:6] == expected_g1)
    assert all(np.asarray(rd.treeID)[6:] == expected_g2)


def test_writer_gpstime_zero_no_tree_get_na_sentinel(
    build_synthetic_las, tmp_path
):
    """tree_id == 0 means "no tree" → output gets the float64 NA sentinel."""
    las = _las_with_gpstime(build_synthetic_las)
    n = len(las.x)
    tree_id = np.zeros(n, dtype=np.int32)
    tree_id[3:6] = 1  # one tree spanning indices 3..5

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="gpstime"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    tiny = np.finfo(np.float64).tiny
    assert all(np.asarray(rd.treeID)[:3] == tiny)
    assert all(np.asarray(rd.treeID)[6:] == tiny)


def test_writer_gpstime_int32_max_sentinel_also_skipped(
    build_synthetic_las, tmp_path
):
    """tree_id == int32.max (LAS NA from out-of-grid merge) is also skipped."""
    las = _las_with_gpstime(build_synthetic_las)
    n = len(las.x)
    tree_id = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    tree_id[:6] = 1

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="gpstime"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    tiny = np.finfo(np.float64).tiny
    assert all(np.asarray(rd.treeID)[6:] == tiny)


def test_writer_gpstime_requires_gpstime_dimension(
    build_synthetic_las, tmp_path
):
    """No gps_time → raise per lidR segment_trees.R:17-18."""
    las = build_synthetic_las(point_format=0)  # format 0 has no gps_time
    n = len(las.x)
    tree_id = np.ones(n, dtype=np.int32)
    with pytest.raises(ValueError, match="gps_time"):
        pylidar.io.write_las_with_treeid(
            las, tree_id, tmp_path / "out.las", uniqueness="gpstime"
        )


def test_writer_gpstime_all_zero_rejected(build_synthetic_las, tmp_path):
    """gpstime all zero → lidR segment_trees.R:20-21 raises."""
    las = _las_with_gpstime(
        build_synthetic_las,
        gpstime_values=np.zeros(12, dtype=np.float64),
    )
    n = len(las.x)
    tree_id = np.ones(n, dtype=np.int32)
    with pytest.raises(ValueError, match="not populated"):
        pylidar.io.write_las_with_treeid(
            las, tree_id, tmp_path / "out.las", uniqueness="gpstime"
        )


def test_writer_gpstime_tie_break_lowest_gpstime(
    build_synthetic_las, tmp_path
):
    """Per lidR tapex: ties on max-z resolve to the lowest gpstime."""
    las = build_synthetic_las()
    n = len(las.x)
    # All same z within a group → tie. Set gps_time deterministically.
    las.z = np.full(n, 5.0)
    las.gps_time = np.array(
        [10.0, 5.0, 7.0, 3.0, 8.0, 2.0, 1.0, 4.0, 9.0, 6.0, 11.0, 12.0]
    )
    tree_id = np.zeros(n, dtype=np.int32)
    tree_id[:6] = 1  # ties on z, expect min gpstime in group = 2.0
    tree_id[6:] = 2  # expect 1.0

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="gpstime"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    out = np.asarray(rd.treeID)
    assert all(out[:6] == 2.0)
    assert all(out[6:] == 1.0)


def test_writer_bitmerge_recomputes_apex_id(build_synthetic_las, tmp_path):
    """Apex (max-z) (x,y) → bitmerge of int32-scaled coords."""
    las = build_synthetic_las()
    n = len(las.x)
    tree_id = np.zeros(n, dtype=np.int32)
    tree_id[:6] = 1
    tree_id[6:] = 2

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="bitmerge"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    assert rd.treeID.dtype == np.float64

    # Recompute the expected bitmerge values by hand.
    z = np.asarray(las.z)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    xoff, yoff, _ = las.header.offsets
    xs, ys, _ = las.header.scales

    apex1 = np.argmax(z[:6])
    apex2 = 6 + np.argmax(z[6:])
    from pylidar.locate_trees import bitmerge
    expect_1 = bitmerge(
        np.array([int((x[apex1] - xoff) / xs)], dtype=np.int32),
        np.array([int((y[apex1] - yoff) / ys)], dtype=np.int32),
    )[0]
    expect_2 = bitmerge(
        np.array([int((x[apex2] - xoff) / xs)], dtype=np.int32),
        np.array([int((y[apex2] - yoff) / ys)], dtype=np.int32),
    )[0]
    out = np.asarray(rd.treeID)
    assert all(out[:6] == expect_1)
    assert all(out[6:] == expect_2)


def test_writer_bitmerge_tie_break_lowest_x(build_synthetic_las, tmp_path):
    """Per lidR xyapex: ties on max-z resolve to the lowest x."""
    las = build_synthetic_las()
    n = len(las.x)
    las.z = np.full(n, 5.0)  # full tie
    tree_id = np.zeros(n, dtype=np.int32)
    tree_id[:6] = 1
    tree_id[6:] = 2

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="bitmerge"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    out = np.asarray(rd.treeID)

    x = np.asarray(las.x)
    y = np.asarray(las.y)
    xoff, yoff, _ = las.header.offsets
    xs, ys, _ = las.header.scales
    # Group 1 (indices 0..5): apex tie-break = argmin(x)
    apex1 = np.argmin(x[:6])
    apex2 = 6 + np.argmin(x[6:])
    from pylidar.locate_trees import bitmerge
    expect_1 = bitmerge(
        np.array([int((x[apex1] - xoff) / xs)], dtype=np.int32),
        np.array([int((y[apex1] - yoff) / ys)], dtype=np.int32),
    )[0]
    expect_2 = bitmerge(
        np.array([int((x[apex2] - xoff) / xs)], dtype=np.int32),
        np.array([int((y[apex2] - yoff) / ys)], dtype=np.int32),
    )[0]
    assert all(out[:6] == expect_1)
    assert all(out[6:] == expect_2)


def test_writer_bitmerge_no_tree_get_na_sentinel(
    build_synthetic_las, tmp_path
):
    las = build_synthetic_las()
    n = len(las.x)
    tree_id = np.zeros(n, dtype=np.int32)
    tree_id[3:6] = 1

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="bitmerge"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    tiny = np.finfo(np.float64).tiny
    out = np.asarray(rd.treeID)
    assert all(out[:3] == tiny)
    assert all(out[6:] == tiny)


def test_writer_float64_input_persists_as_is_under_bitmerge_uniqueness(
    build_synthetic_las, tmp_path
):
    """When tree_id is already float64 (Phase-4 propagation), the writer
    treats it as already-finalized and persists without recomputation."""
    las = build_synthetic_las()
    n = len(las.x)
    tree_id = np.array([float(i) * 1.5 for i in range(n)], dtype=np.float64)

    pylidar.io.write_las_with_treeid(
        las, tree_id, tmp_path / "out.las", uniqueness="bitmerge"
    )
    rd = laspy.read(str(tmp_path / "out.las"))
    assert list(np.asarray(rd.treeID)) == list(tree_id)
