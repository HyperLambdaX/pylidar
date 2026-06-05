"""End-to-end tests for the lidR-aligned Dalponte demo."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
TOOLS_DIR = REPO_ROOT / "tools"


@pytest.fixture(scope="module")
def dalponte_demo_module():
    spec = importlib.util.spec_from_file_location(
        "_dalponte2016_demo", EXAMPLES_DIR / "dalponte2016_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dalponte2016_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen_module():
    spec = importlib.util.spec_from_file_location(
        "_gen_synthetic_las_dalponte", TOOLS_DIR / "gen_synthetic_las.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_synthetic_las_dalponte"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dalponte_demo_end_to_end_default(
    dalponte_demo_module, gen_module, tmp_path
):
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=4,
        n_points_per_tree=180,
        n_ground=120,
        point_format=3,
        seed=33,
    )
    out_dir = tmp_path / "out"
    dalponte_demo_module.main(in_path, out_dir)

    out_path = out_dir / "dalponte2016.las"
    assert out_path.exists()
    las_in = laspy.read(str(in_path))
    las_out = laspy.read(str(out_path))
    assert len(las_out.x) == len(las_in.x)
    assert las_out.point_format.id == las_in.point_format.id
    assert "treeID" in las_out.point_format.dimension_names
    tree_id = np.asarray(las_out.treeID)
    assert tree_id.dtype == np.int32
    assert float((tree_id > 0).mean()) >= 0.5

    # Viewer-friendly RGB companion: RGB-capable format, >2 distinct colours.
    rgb_path = out_dir / "dalponte2016_rgb.las"
    assert rgb_path.exists()
    rgb_las = laspy.read(str(rgb_path))
    assert int(rgb_las.point_format.id) in {2, 3, 5, 7, 8, 10}
    colors = np.stack([np.asarray(rgb_las.red), np.asarray(rgb_las.green),
                       np.asarray(rgb_las.blue)], axis=1)
    assert len(np.unique(colors, axis=0)) >= 3


def test_dalponte_demo_exports_chm_and_treetops(
    dalponte_demo_module, gen_module, tmp_path
):
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=3,
        n_points_per_tree=160,
        n_ground=100,
        point_format=3,
        seed=44,
    )
    out_dir = tmp_path / "out"
    dalponte_demo_module.main(
        in_path,
        out_dir,
        export_chm=True,
        export_treetops=True,
    )

    raw_chm = np.load(out_dir / "chm_raw.npy")
    smoothed_chm = np.load(out_dir / "chm.npy")
    assert raw_chm.shape == smoothed_chm.shape
    sidecar = json.loads((out_dir / "chm.json").read_text())
    assert sidecar["shape"] == list(smoothed_chm.shape)
    assert sidecar["params"]["res"] == 0.5
    assert sidecar["params"]["subcircle"] == 0.3
    assert sidecar["params"]["focal_size"] == 3
    assert sidecar["params"]["ws"] == 4.0
    assert sidecar["params"]["hmin"] == 2.0
    assert sidecar["params"]["normalize"] is True
    assert sidecar["params"]["ground_algorithm"] == "csf"
    assert sidecar["params"]["ground_candidate_source"] == "last_returns"
    assert sidecar["params"]["ground_points"] > 0
    assert sidecar["params"]["csf_class_threshold"] == 0.5
    assert sidecar["params"]["csf_cloth_resolution"] == 0.5
    assert sidecar["params"]["csf_rigidness"] == 1
    assert sidecar["params"]["normalize_method"] == "tin"
    assert sidecar["params"]["mask_treeid_z_below"] == 0.0
    assert "normalize_fallback_counts" in sidecar["params"]

    csv_path = out_dir / "treetops_dalponte2016.csv"
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0] == "x,y,z,tree_id"
    assert len(rows) >= 2
    x, y, z, tid = rows[1].split(",")
    float(x); float(y); float(z); int(tid)


def test_dalponte_demo_bitmerge_writes_double_treeid(
    dalponte_demo_module, gen_module, tmp_path
):
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=3,
        n_points_per_tree=160,
        n_ground=100,
        point_format=3,
        seed=55,
    )
    out_dir = tmp_path / "out"
    dalponte_demo_module.main(in_path, out_dir, uniqueness="bitmerge")

    las_out = laspy.read(str(out_dir / "dalponte2016.las"))
    assert "treeID" in las_out.point_format.dimension_names
    assert np.asarray(las_out.treeID).dtype == np.float64


def test_dalponte_demo_no_normalize_metadata(
    dalponte_demo_module, gen_module, tmp_path
):
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=3,
        n_points_per_tree=160,
        n_ground=100,
        point_format=3,
        seed=66,
    )
    out_dir = tmp_path / "out"
    dalponte_demo_module.main(
        in_path,
        out_dir,
        normalize=False,
        export_chm=True,
    )

    sidecar = json.loads((out_dir / "chm.json").read_text())
    assert sidecar["params"]["normalize"] is False
    assert sidecar["params"]["ground_algorithm"] is None
    assert sidecar["params"]["normalize_method"] is None
    assert sidecar["params"]["mask_treeid_z_below"] is None


def test_dalponte_demo_pmf_ground_method_metadata(
    dalponte_demo_module, gen_module, tmp_path
):
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=3,
        n_points_per_tree=160,
        n_ground=100,
        point_format=3,
        seed=67,
    )
    out_dir = tmp_path / "out"
    dalponte_demo_module.main(
        in_path,
        out_dir,
        ground_method="pmf",
        export_chm=True,
    )

    sidecar = json.loads((out_dir / "chm.json").read_text())
    assert sidecar["params"]["normalize"] is True
    assert sidecar["params"]["ground_algorithm"] == "pmf"
    assert sidecar["params"]["ground_ws"] == [5.0, 9.0, 13.0, 17.0]
    assert sidecar["params"]["ground_th"] == [3.0, 3.0, 3.0, 3.0]


def test_dalponte_demo_masks_low_normalized_treeids_when_requested(
    dalponte_demo_module, gen_module, tmp_path
):
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=3,
        n_points_per_tree=160,
        n_ground=100,
        point_format=3,
        seed=77,
    )
    out_dir = tmp_path / "out"
    dalponte_demo_module.main(in_path, out_dir, mask_treeid_z_below=0.5)

    las_in = laspy.read(str(in_path))
    las_out = laspy.read(str(out_dir / "dalponte2016.las"))
    ground_like = np.asarray(las_in.classification) == 2
    tree_id = np.asarray(las_out.treeID)
    na = np.iinfo(tree_id.dtype).max
    assert np.all(tree_id[ground_like] == na)


def _assert_corridor_segmentation_is_sane(out_dir, in_path, expected_algo):
    """Shared checks: the negative-elevation corridor pipeline must not collapse.

    Before the negative-Z fixes, ground classification on negative absolute
    elevations misbehaved (PMF's all-negative-window quirk classified every
    point as ground), the TIN surface then absorbed the canopy, and almost no
    point received a treeID. These assertions pin the corrected behaviour.
    """
    sidecar = json.loads((out_dir / "chm.json").read_text())
    params = sidecar["params"]
    assert params["normalize"] is True
    assert params["ground_algorithm"] == expected_algo
    # Ground must be a non-trivial *subset* — not "all points are ground".
    n_in = len(laspy.read(str(in_path)).x)
    assert 0 < params["ground_points"] < n_in

    las_out = laspy.read(str(out_dir / "dalponte2016.las"))
    tree_id = np.asarray(las_out.treeID)
    na = np.iinfo(tree_id.dtype).max
    assigned = (tree_id > 0) & (tree_id != na)
    # A healthy run assigns a real crown to a meaningful share of points.
    assert float(assigned.mean()) >= 0.3
    # At least two distinct trees recovered on a 3-tree scene.
    assert len(np.unique(tree_id[assigned])) >= 2


def test_dalponte_demo_negative_elevation_corridor_csf(
    dalponte_demo_module, gen_module, tmp_path
):
    # Tilted terrain ~ -22 .. -6 m absolute (the 10kV-corridor regime).
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=3,
        n_points_per_tree=200,
        n_ground=250,
        area=30.0,
        point_format=3,
        seed=101,
        z_offset=-22.0,
        ground_slope=0.2,
    )
    out_dir = tmp_path / "out"
    # CSF is the demo default; assert it natively handles negative elevations.
    dalponte_demo_module.main(in_path, out_dir, ground_method="csf", export_chm=True)
    _assert_corridor_segmentation_is_sane(out_dir, in_path, "csf")


def test_dalponte_demo_negative_elevation_corridor_pmf_shift(
    dalponte_demo_module, gen_module, tmp_path
):
    # Same negative-elevation scene through PMF: exercises the Python-layer
    # positive-Z shift that neutralizes lidR's all-negative-window quirk.
    in_path = gen_module.make_forest_las(
        tmp_path / "forest.las",
        n_trees=3,
        n_points_per_tree=200,
        n_ground=250,
        area=30.0,
        point_format=3,
        seed=101,
        z_offset=-22.0,
        ground_slope=0.2,
    )
    out_dir = tmp_path / "out"
    dalponte_demo_module.main(in_path, out_dir, ground_method="pmf", export_chm=True)
    _assert_corridor_segmentation_is_sane(out_dir, in_path, "pmf")
