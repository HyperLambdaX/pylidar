"""End-to-end tests for the li2012 point-based ITS demo."""

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
def li_demo():
    spec = importlib.util.spec_from_file_location(
        "_li2012_demo", EXAMPLES_DIR / "li2012_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_li2012_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen_module():
    spec = importlib.util.spec_from_file_location(
        "_gen_synthetic_las_li", TOOLS_DIR / "gen_synthetic_las.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_synthetic_las_li"] = mod
    spec.loader.exec_module(mod)
    return mod


def _forest(gen_module, tmp_path, **kw):
    defaults = dict(
        n_trees=4, n_points_per_tree=160, n_ground=120, point_format=3, seed=33
    )
    defaults.update(kw)
    return gen_module.make_forest_las(tmp_path / "forest.las", **defaults)


def test_li2012_demo_end_to_end_default(li_demo, gen_module, tmp_path):
    in_path = _forest(gen_module, tmp_path)
    out_dir = tmp_path / "out"
    li_demo.main(in_path, out_dir, export_treetops=True)

    out_path = out_dir / "li2012.las"
    assert out_path.exists()
    las_in = laspy.read(str(in_path))
    las_out = laspy.read(str(out_path))
    # treeID is written back to the *full* original cloud.
    assert len(las_out.x) == len(las_in.x)
    assert "treeID" in las_out.point_format.dimension_names
    tree_id = np.asarray(las_out.treeID)
    assert tree_id.dtype == np.int32
    assert float((tree_id > 0).mean()) >= 0.5

    sidecar = json.loads((out_dir / "li2012.json").read_text())
    assert sidecar["n_trees"] >= 1
    assert sidecar["params"]["hmin"] == 2.0
    assert sidecar["normalize"] is True
    assert sidecar["ground_algorithm"] == "csf"

    csv_path = out_dir / "treetops_li2012.csv"
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0] == "x,y,z,tree_id"
    assert len(rows) - 1 == sidecar["n_trees"]

    # Viewer-friendly RGB companion exists, upgrades to an RGB format, and
    # actually carries more than one colour (per-tree palette, not 2 bins).
    rgb_path = out_dir / "li2012_rgb.las"
    assert rgb_path.exists()
    rgb_las = laspy.read(str(rgb_path))
    assert int(rgb_las.point_format.id) in {2, 3, 5, 7, 8, 10}
    colors = np.stack([np.asarray(rgb_las.red), np.asarray(rgb_las.green),
                       np.asarray(rgb_las.blue)], axis=1)
    assert len(np.unique(colors, axis=0)) >= 3


def test_li2012_demo_negative_elevation_corridor(li_demo, gen_module, tmp_path):
    # Tilted, negative-elevation terrain (the MLS-corridor regime).
    in_path = _forest(
        gen_module, tmp_path, n_trees=3, n_points_per_tree=200, n_ground=200,
        area=30.0, seed=101, z_offset=-22.0, ground_slope=0.2,
    )
    out_dir = tmp_path / "out"
    li_demo.main(in_path, out_dir)

    sidecar = json.loads((out_dir / "li2012.json").read_text())
    n_in = len(laspy.read(str(in_path)).x)
    assert 0 < sidecar["ground_points"] < n_in
    assert sidecar["n_trees"] >= 2

    las_out = laspy.read(str(out_dir / "li2012.las"))
    tree_id = np.asarray(las_out.treeID)
    assert float((tree_id > 0).mean()) >= 0.3


def test_li2012_demo_voxel_downsample_path(li_demo, gen_module, tmp_path):
    in_path = _forest(gen_module, tmp_path, n_trees=4, n_points_per_tree=300)
    out_dir = tmp_path / "out"
    # Force the voxel downsample + label-propagation branch.
    li_demo.main(in_path, out_dir, target_points=120)

    sidecar = json.loads((out_dir / "li2012.json").read_text())
    assert sidecar["segment_points"] <= 200
    assert sidecar["voxel"] > 0.0
    # Propagation still assigns every original point.
    las_out = laspy.read(str(out_dir / "li2012.las"))
    assert len(las_out.x) == len(laspy.read(str(in_path)).x)


def test_li2012_demo_crop_and_max_points_guard(li_demo, gen_module, tmp_path):
    in_path = _forest(gen_module, tmp_path, n_trees=4, n_points_per_tree=300)
    out_dir = tmp_path / "out"
    # voxel=tiny + low max_points -> O(N^2) guard must trip.
    with pytest.raises(SystemExit, match="max-points"):
        li_demo.main(in_path, out_dir, voxel=0.01, max_points=50)


def test_li2012_demo_no_normalize_runs_on_raw_heights(li_demo, gen_module, tmp_path):
    in_path = _forest(gen_module, tmp_path)
    out_dir = tmp_path / "out"
    li_demo.main(in_path, out_dir, normalize=False)

    sidecar = json.loads((out_dir / "li2012.json").read_text())
    assert sidecar["normalize"] is False
    assert sidecar["ground_algorithm"] is None
    las_out = laspy.read(str(out_dir / "li2012.las"))
    assert "treeID" in las_out.point_format.dimension_names
