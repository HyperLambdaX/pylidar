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
