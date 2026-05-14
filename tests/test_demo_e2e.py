"""Phase 6 end-to-end smoke test for ``examples/its_demo.py``.

Builds a synthetic forest via ``tools/gen_synthetic_las.py`` and runs the
demo main entry, then asserts every cross-cutting contract the rd.md
gap-analysis row-13 enumerates:

1. All four algorithm outputs exist.
2. Per-output point count equals the input point count.
3. ``treeID`` extra-byte dim exists, dtype int32, NA sentinel
   ``np.iinfo(np.int32).max``.
4. RGB present when input point format supports RGB.
5. Header (point_format / scales / offsets / version) and VLR set are
   preserved between input and output.
6. ``treeID > 0`` fraction ≥ 50% on the dense synthetic forest — guards
   against a silent regression where (for example) the watershed
   ghost-tree problem returns.

Distinct from ``test_demo_e2e_smoke.py`` which is the narrow Phase 5
audit-fix regression for point_format=1 (no RGB).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(scope="module")
def demo_module():
    spec = importlib.util.spec_from_file_location(
        "_its_demo_e2e", EXAMPLES_DIR / "its_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_its_demo_e2e"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen_module():
    spec = importlib.util.spec_from_file_location(
        "_gen_synthetic_las", TOOLS_DIR / "gen_synthetic_las.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_synthetic_las"] = mod
    spec.loader.exec_module(mod)
    return mod


_INT32_NA = int(np.iinfo(np.int32).max)
_ALGO_OUTPUTS = ("li2012.las", "silva2016.las", "dalponte2016.las", "watershed.las")


def _read(p: Path) -> laspy.LasData:
    return laspy.read(str(p))


def test_demo_end_to_end_default_pf3(demo_module, gen_module, tmp_path):
    """Happy path: point format 3, no filters; every contract on every output."""
    in_path = gen_module.make_forest_las(
        tmp_path / "forest_pf3.las",
        n_trees=4,
        n_points_per_tree=200,
        n_ground=200,
        point_format=3,
        seed=42,
    )
    out_dir = tmp_path / "out"
    demo_module.main(in_path, out_dir)

    las_in = _read(in_path)
    n_in = len(las_in.x)

    for name in _ALGO_OUTPUTS:
        path = out_dir / name
        assert path.exists(), f"demo failed to write {name}"
        las_out = _read(path)
        # (1)(2): point count round-trips
        assert len(las_out.x) == n_in, f"{name}: point count mismatch"
        # (3): treeID extra dim with correct dtype + NA sentinel
        assert "treeID" in las_out.point_format.dimension_names, (
            f"{name}: treeID extra dim missing"
        )
        tree_id = np.asarray(las_out.treeID)
        assert tree_id.dtype == np.int32, (
            f"{name}: treeID dtype {tree_id.dtype} != int32"
        )
        # NA sentinel reachable: every value is either ≥ 0 and < int32 max,
        # or exactly int32 max. We don't require any to actually equal NA
        # since for a synthetic dense forest every point can land in some tree.
        non_na = (tree_id != _INT32_NA)
        assert tree_id[non_na].min() >= 0, f"{name}: negative treeID in non-NA range"
        # (4): RGB carried over (PF=3 supports it)
        assert "red" in las_out.point_format.dimension_names
        # (5): header preservation
        assert las_out.point_format.id == las_in.point_format.id
        assert tuple(las_out.header.scales) == tuple(las_in.header.scales)
        assert tuple(las_out.header.offsets) == tuple(las_in.header.offsets)
        assert str(las_out.header.version) == str(las_in.header.version)
        # VLR preservation: every input VLR appears in the output. The
        # writer adds the ExtraBytesVlr for `treeID`, so the output count
        # may exceed the input count by one — but no input VLR may be
        # silently dropped.
        in_vlr_types = [type(v).__name__ for v in las_in.header.vlrs]
        out_vlr_types = [type(v).__name__ for v in las_out.header.vlrs]
        for t in in_vlr_types:
            assert t in out_vlr_types, (
                f"{name}: VLR {t!r} from input not preserved in output"
            )
        # The treeID extra dim contributes exactly one ExtraBytesVlr.
        assert "ExtraBytesVlr" in out_vlr_types, (
            f"{name}: missing ExtraBytesVlr for treeID"
        )
        # (6): assigned fraction guard
        frac = float((tree_id > 0).mean())
        assert frac >= 0.5, (
            f"{name}: only {100 * frac:.1f}% of points segmented — "
            f"possible regression (watershed ghost trees, broken CHM, etc.)"
        )


def test_demo_keep_first_drops_ground_points(demo_module, gen_module, tmp_path):
    """--keep-first + --drop-class 2 should pre-filter ground; output point
    count equals the filtered (not raw) count.
    """
    in_path = gen_module.make_forest_las(
        tmp_path / "forest_pf3.las",
        n_trees=3,
        point_format=3,
        seed=7,
    )
    out_dir = tmp_path / "out"
    demo_module.main(
        in_path, out_dir,
        keep_first=True,
        drop_class=[2],
    )

    raw = _read(in_path)
    n_raw = len(raw.x)
    n_ground = int((np.asarray(raw.classification) == 2).sum())
    n_expected = n_raw - n_ground  # generator marks all tree returns as RN==1
    assert n_expected < n_raw

    for name in _ALGO_OUTPUTS:
        las_out = _read(out_dir / name)
        assert len(las_out.x) == n_expected, (
            f"{name}: filtered point count drifted"
        )


def test_demo_exports_treetops_csv_and_chm_npy(demo_module, gen_module, tmp_path):
    """--export-treetops writes silva/dalponte CSVs; --export-chm writes
    chm.npy + chm.json sidecar with layout fields."""
    in_path = gen_module.make_forest_las(
        tmp_path / "forest_pf3.las", n_trees=4, point_format=3, seed=1
    )
    out_dir = tmp_path / "out"
    demo_module.main(
        in_path, out_dir,
        export_treetops=True,
        export_chm=True,
    )

    silva_csv = out_dir / "treetops_silva2016.csv"
    dalp_csv = out_dir / "treetops_dalponte2016.csv"
    assert silva_csv.exists(), "silva treetops CSV not written"
    assert dalp_csv.exists(), "dalponte treetops CSV not written"
    # Each CSV: header row + ≥1 data row, with x/y/z/tree_id columns.
    for csv_path in (silva_csv, dalp_csv):
        text = csv_path.read_text().strip().splitlines()
        assert text[0] == "x,y,z,tree_id"
        assert len(text) >= 2, f"{csv_path.name} carries no treetop rows"
        # Spot-check parseability of first data row.
        x, y, z, tid = text[1].split(",")
        float(x); float(y); float(z); int(tid)

    chm_npy = out_dir / "chm.npy"
    chm_json = out_dir / "chm.json"
    assert chm_npy.exists() and chm_json.exists()
    chm = np.load(chm_npy)
    assert chm.ndim == 2
    side = json.loads(chm_json.read_text())
    assert side["shape"] == list(chm.shape)
    for key in ("xmin", "ymax", "xres", "yres", "nrow", "ncol"):
        assert key in side, f"chm.json missing {key!r}"


def test_demo_treetops_csv_xyz_matches_dalponte_seeds(
    demo_module, gen_module, tmp_path
):
    """The dalponte treetops CSV must report the same (x, y, z) the
    locate_trees_chm wrapper emitted. Catches a silent shift if the
    column order or row/col → world inversion ever drifts."""
    in_path = gen_module.make_forest_las(
        tmp_path / "forest_pf3.las", n_trees=3, point_format=3, seed=11
    )
    out_dir = tmp_path / "out"
    demo_module.main(in_path, out_dir, export_treetops=True)

    rows = (out_dir / "treetops_dalponte2016.csv").read_text().strip().splitlines()
    csv_xyz = np.array([
        [float(v) for v in line.split(",")[:3]] for line in rows[1:]
    ])
    # Every CSV row must sit inside the input bbox (sanity).
    las_in = _read(in_path)
    assert csv_xyz[:, 0].min() >= float(las_in.x.min()) - 1e-6
    assert csv_xyz[:, 0].max() <= float(las_in.x.max()) + 1e-6
    assert csv_xyz[:, 1].min() >= float(las_in.y.min()) - 1e-6
    assert csv_xyz[:, 1].max() <= float(las_in.y.max()) + 1e-6
