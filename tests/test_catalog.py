"""Tests for ``pylidar.catalog`` (Phase 6).

Covers the four test cases enumerated in ``task_plan.md`` Phase 6:

1. Single-tile run with ``n_workers=1``.
2. Multi-tile chain — two adjacent tiles processed back-to-back.
3. Cross-tile tree-ID uniqueness with ``uniqueness="bitmerge"``.
4. Buffer seam: a tile's neighbor list correctly identifies the
   spatially adjacent tile and ``core_mask`` slices the buffer ring
   so each input point appears in exactly one tile's output.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

import pylidar
import pylidar.catalog as pcat

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gen_module():
    spec = importlib.util.spec_from_file_location(
        "_gen_synthetic_las_cat",
        REPO_ROOT / "tools" / "gen_synthetic_las.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_synthetic_las_cat"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_two_tile_catalog(
    gen_module, tmp_path: Path
) -> tuple[pcat.LAScatalog, Path, Path]:
    """Two adjacent 25 m × 25 m tiles laid out side-by-side along X."""
    t1 = gen_module.make_forest_las(
        tmp_path / "tile_west.las",
        n_trees=3, point_format=3, seed=11, area=25.0,
    )
    t2_path = tmp_path / "tile_east.las"
    # The east tile is the same forest translated 25 m east.
    gen_module.make_forest_las(
        t2_path, n_trees=3, point_format=3, seed=22, area=25.0,
    )
    # Patch east tile xs by reading + adding 25 m, then re-writing.
    las = laspy.read(str(t2_path))
    las.x = np.asarray(las.x) + 25.0
    las.write(str(t2_path))
    catalog = pcat.LAScatalog([t1, t2_path], buffer=3.0)
    return catalog, t1, t2_path


# ---------------------------------------------------------------- construction

def test_catalog_construction_validates_files(tmp_path):
    with pytest.raises(ValueError):
        pcat.LAScatalog([])

    bogus = tmp_path / "does_not_exist.las"
    with pytest.raises(FileNotFoundError):
        pcat.LAScatalog([bogus])


def test_catalog_negative_buffer_rejected(gen_module, tmp_path):
    p = gen_module.make_forest_las(tmp_path / "t.las", n_trees=2, seed=0)
    with pytest.raises(ValueError):
        pcat.LAScatalog([p], buffer=-1.0)


def test_catalog_tile_bounds_from_header(gen_module, tmp_path):
    p = gen_module.make_forest_las(tmp_path / "t.las", n_trees=3, seed=1, area=20.0)
    cat = pcat.LAScatalog([p])
    assert cat.n_tiles == 1
    bnds = cat.tile_bounds[0]
    assert len(bnds) == 4
    # Approx 0 .. 20 m area (with some jitter from random tree placement).
    assert bnds[0] >= -2.0 and bnds[2] <= 25.0
    assert bnds[1] >= -2.0 and bnds[3] <= 25.0


# ---------------------------------------------------------------- neighbors

def test_neighbors_empty_with_zero_buffer(gen_module, tmp_path):
    cat, _, _ = _make_two_tile_catalog(gen_module, tmp_path)
    cat0 = pcat.LAScatalog(list(cat.files), buffer=0.0)
    assert cat0.neighbors_of(0) == ()
    assert cat0.neighbors_of(1) == ()


def test_neighbors_detect_adjacent_tile(gen_module, tmp_path):
    cat, t1, t2 = _make_two_tile_catalog(gen_module, tmp_path)
    n0 = cat.neighbors_of(0)
    n1 = cat.neighbors_of(1)
    assert t2 in n0
    assert t1 in n1


def test_neighbors_skip_far_tiles(gen_module, tmp_path):
    p1 = gen_module.make_forest_las(tmp_path / "a.las", seed=1)
    # Place a far tile 200 m east.
    p2 = tmp_path / "b.las"
    gen_module.make_forest_las(p2, seed=2)
    las = laspy.read(str(p2))
    las.x = np.asarray(las.x) + 200.0
    las.write(str(p2))
    cat = pcat.LAScatalog([p1, p2], buffer=5.0)
    assert cat.neighbors_of(0) == ()
    assert cat.neighbors_of(1) == ()


# ---------------------------------------------------------------- map_tiles

def test_map_tiles_single_worker_returns_per_tile(gen_module, tmp_path):
    cat, _, _ = _make_two_tile_catalog(gen_module, tmp_path)
    results = cat.map_tiles(lambda ctx: ctx.tile_index, n_workers=1)
    assert results == [0, 1]


def test_map_tiles_passes_correct_context(gen_module, tmp_path):
    cat, t1, t2 = _make_two_tile_catalog(gen_module, tmp_path)
    paths = cat.map_tiles(lambda ctx: ctx.tile_path, n_workers=1)
    assert paths == [t1, t2]


# ---------------------------------------------------------------- buffered load

def test_load_no_buffer_returns_only_tile_points(gen_module, tmp_path):
    cat, _, _ = _make_two_tile_catalog(gen_module, tmp_path)
    no_buf = pcat.LAScatalog(list(cat.files), buffer=0.0)
    ctx = no_buf.make_context(0)
    las, core_mask = ctx.load()
    assert core_mask.all(), "core mask should be all-True when no buffer is loaded"
    assert len(las.x) == len(laspy.read(str(no_buf.files[0])).x)


def test_load_with_buffer_pulls_neighbor_points(gen_module, tmp_path):
    cat, _, _ = _make_two_tile_catalog(gen_module, tmp_path)
    ctx = cat.make_context(0)
    las_buf, core_mask = ctx.load()
    n_original = len(laspy.read(str(cat.files[0])).x)
    # Some points should fall outside the unbuffered tile bbox (= the
    # neighbor's buffer ring).
    assert not core_mask.all(), "buffered load did not bring in neighbor points"
    # Original-tile points come first in `_concat_las`; the first N
    # entries of core_mask correspond to original-tile points (all True).
    assert core_mask[:n_original].all()


def test_core_mask_filters_to_unbuffered_bbox(gen_module, tmp_path):
    cat, _, _ = _make_two_tile_catalog(gen_module, tmp_path)
    ctx = cat.make_context(0)
    las_buf, core_mask = ctx.load()
    xs = np.asarray(las_buf.x)
    ys = np.asarray(las_buf.y)
    xmin, ymin, xmax, ymax = ctx.tile_bounds
    expected = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    np.testing.assert_array_equal(core_mask, expected)


# ---------------------------------------------------------------- segment_trees_catalog

def _toy_locate(las):
    # Trivial locator: every point with z > 8 is a "treetop".
    return np.asarray(las.z, dtype=np.float64) > 8.0


def _toy_segment(las, mask):
    # Label every point with the index of the nearest treetop in xy.
    tx = np.asarray(las.x)[mask]
    ty = np.asarray(las.y)[mask]
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    if tx.size == 0:
        return np.zeros(len(x), dtype=np.int32)
    dx = x[:, None] - tx[None, :]
    dy = y[:, None] - ty[None, :]
    d = dx * dx + dy * dy
    return (np.argmin(d, axis=1) + 1).astype(np.int32)


def test_segment_trees_catalog_rejects_incremental(gen_module, tmp_path):
    cat, _, _ = _make_two_tile_catalog(gen_module, tmp_path)
    with pytest.raises(ValueError, match="incremental"):
        pcat.segment_trees_catalog(
            cat,
            locate_fn=_toy_locate,
            segment_fn=_toy_segment,
            output_dir=tmp_path / "out",
            uniqueness="incremental",
        )


def test_segment_trees_catalog_writes_per_tile_output(gen_module, tmp_path):
    cat, t1, t2 = _make_two_tile_catalog(gen_module, tmp_path)
    out_dir = tmp_path / "out"
    paths = pcat.segment_trees_catalog(
        cat,
        locate_fn=_toy_locate,
        segment_fn=_toy_segment,
        output_dir=out_dir,
        uniqueness="bitmerge",
    )
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    for p, src in zip(paths, (t1, t2)):
        # Point count equals the source tile's count (no buffer leakage
        # into output).
        out_las = laspy.read(str(p))
        in_las = laspy.read(str(src))
        assert len(out_las.x) == len(in_las.x)
        assert "treeID" in out_las.point_format.dimension_names
        # bitmerge → float64 treeID
        assert np.asarray(out_las.treeID).dtype == np.float64


def test_map_tiles_multiworker_raises_without_joblib(gen_module, tmp_path):
    """When joblib isn't installed, ``n_workers > 1`` must raise a
    friendly :class:`ImportError` (Q7 decision; the catalog extra is
    not in the base deps)."""
    try:
        import joblib  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("joblib is installed; skip the missing-extra path")
    cat, _, _ = _make_two_tile_catalog(gen_module, tmp_path)
    with pytest.raises(ImportError, match="pylidar\\[catalog\\]"):
        cat.map_tiles(lambda ctx: ctx.tile_index, n_workers=2)


def _make_seam_straddling_tile(
    path: Path, x_min: float, x_max: float, apex_x: float, *, seed: int
) -> None:
    """Two tile layouts share the same physical tree centred at ``apex_x``;
    each tile keeps only the points within its own x range. Designed for
    :func:`test_catalog_apex_uniqueness_crosses_tile_seam`.
    """
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0, 2 * np.pi, 300)
    radii = rng.uniform(0, 3.0, 300)
    x = apex_x + radii * np.cos(thetas)
    y = 10.0 + radii * np.sin(thetas)
    z = 14.0 - radii + rng.normal(0, 0.2, 300)
    inside = (x >= x_min) & (x < x_max)
    x, y, z = x[inside], y[inside], z[inside]
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales = (0.001, 0.001, 0.001)
    hdr.offsets = (0.0, 0.0, 0.0)
    las = laspy.LasData(hdr)
    las.x = x
    las.y = y
    las.z = z
    las.return_number = np.ones(len(x), np.uint8)
    las.number_of_returns = np.ones(len(x), np.uint8)
    las.classification = np.full(len(x), 5, np.uint8)
    las.intensity = np.zeros(len(x), np.uint16)
    las.gps_time = np.arange(len(x), dtype=np.float64) * 1e-3
    las.red = np.zeros(len(x), np.uint16)
    las.green = np.zeros(len(x), np.uint16)
    las.blue = np.zeros(len(x), np.uint16)
    las.write(str(path))


def _single_tree_locate(las):
    """Locator that returns exactly one treetop: the global max-z point."""
    z = np.asarray(las.z)
    mask = np.zeros(len(z), dtype=bool)
    if z.size:
        mask[int(np.argmax(z))] = True
    return mask


def _radial_segment(las, mask):
    """All points within radius 5 of the single treetop get label 1, else 0."""
    tx = np.asarray(las.x)[mask]
    ty = np.asarray(las.y)[mask]
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    out = np.zeros(len(x), dtype=np.int32)
    if tx.size == 0:
        return out
    d2 = (x - tx[0]) ** 2 + (y - ty[0]) ** 2
    out[d2 <= 25.0] = 1
    return out


def test_catalog_apex_uniqueness_crosses_tile_seam(tmp_path):
    """Regression for the high-severity bug: a tree spanning a tile seam
    must receive the **same** float64 bitmerge ID from both workers.

    Pre-fix the writer re-derived apex from core points only, so each
    side picked its own local max and the bitmerge IDs differed. Post-fix
    apex is computed on the buffered cloud, so both workers see the same
    global apex coordinates.

    Scenario: a single physical tree centred at ``apex_x = 22`` (= tile
    B's core, since B starts at x = 20). With buffer = 8, both workers
    see the full tree in their buffered clouds. Locate picks the
    global max-z; segment labels every within-radius-5 point as
    tree 1. Apex IDs derived on the buffered cloud are identical;
    pre-fix they differ because each side's writer only saw its
    half of the tree.
    """
    apex_x = 22.0
    t_a = tmp_path / "tile_a.las"
    t_b = tmp_path / "tile_b.las"
    _make_seam_straddling_tile(t_a, 0.0, 20.0, apex_x, seed=1)
    _make_seam_straddling_tile(t_b, 20.0, 40.0, apex_x, seed=1)

    cat = pcat.LAScatalog([t_a, t_b], buffer=8.0)
    out_dir = tmp_path / "out"
    paths = pcat.segment_trees_catalog(
        cat,
        locate_fn=_single_tree_locate,
        segment_fn=_radial_segment,
        output_dir=out_dir,
        uniqueness="bitmerge",
    )
    na = float(np.finfo(np.float64).tiny)
    ids_a = np.asarray(laspy.read(str(paths[0])).treeID, dtype=np.float64)
    ids_b = np.asarray(laspy.read(str(paths[1])).treeID, dtype=np.float64)
    set_a = set(ids_a[ids_a != na].tolist())
    set_b = set(ids_b[ids_b != na].tolist())
    assert set_a == {next(iter(set_a))}, (
        f"tile A should carry one bitmerge ID for the single tree, got {set_a}"
    )
    assert set_b == {next(iter(set_b))}, (
        f"tile B should carry one bitmerge ID for the single tree, got {set_b}"
    )
    # The single tree's ID must be the same on both sides — its apex
    # (x, y) was found on the buffered cloud, identical in both workers.
    assert set_a == set_b, (
        "apex bitmerge ID drifted between tile A and tile B for the same "
        f"tree. left: {sorted(set_a)}, right: {sorted(set_b)}"
    )


def test_catalog_rejects_duplicate_input_stems(gen_module, tmp_path):
    """Two input tiles that share a file stem would produce colliding
    output filenames; the catalog must refuse upfront so no result is
    silently overwritten."""
    d1 = tmp_path / "dir_a"
    d2 = tmp_path / "dir_b"
    d1.mkdir()
    d2.mkdir()
    p1 = gen_module.make_forest_las(d1 / "tile_001.las", n_trees=2, seed=1)
    p2 = gen_module.make_forest_las(d2 / "tile_001.las", n_trees=2, seed=2)
    # Shift the second tile so the catalog itself is internally consistent
    # (different bounds), but the stems match.
    las = laspy.read(str(p2))
    las.x = np.asarray(las.x) + 200.0
    las.write(str(p2))

    cat = pcat.LAScatalog([p1, p2], buffer=0.0)
    with pytest.raises(ValueError, match="share file-stem"):
        pcat.segment_trees_catalog(
            cat,
            locate_fn=_toy_locate,
            segment_fn=_toy_segment,
            output_dir=tmp_path / "out",
            uniqueness="bitmerge",
        )


def test_catalog_bitmerge_ids_are_globally_unique(gen_module, tmp_path):
    """Two adjacent tiles run independently; their non-NA treeID sets
    must be disjoint (the bitmerge encoding of apex coordinates is
    globally unique without any cross-worker coordination)."""
    cat, t1, t2 = _make_two_tile_catalog(gen_module, tmp_path)
    out_dir = tmp_path / "out"
    paths = pcat.segment_trees_catalog(
        cat,
        locate_fn=_toy_locate,
        segment_fn=_toy_segment,
        output_dir=out_dir,
        uniqueness="bitmerge",
    )
    na = float(np.finfo(np.float64).tiny)
    ids_a = np.asarray(laspy.read(str(paths[0])).treeID, dtype=np.float64)
    ids_b = np.asarray(laspy.read(str(paths[1])).treeID, dtype=np.float64)
    set_a = set(ids_a[ids_a != na].tolist())
    set_b = set(ids_b[ids_b != na].tolist())
    assert set_a and set_b, "every tile should yield at least one tree"
    # The two tiles do not share trees → bitmerge IDs are disjoint.
    assert set_a.isdisjoint(set_b), (
        "bitmerge tree-IDs collided across tiles — apex-based uniqueness broke"
    )
