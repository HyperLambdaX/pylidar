"""Phase 5 audit-fix #2 (2026-05-12) regression smoke test.

Builds a tiny synthetic point-cloud with point_format=1 (gps_time, no RGB
channels) and runs the demo end-to-end. Pre-audit, the first writer call
would raise ``ValueError: rgb supplied but input point_format 1 has no
RGB channels``. After the audit fix the demo now produces all four
treeID-only outputs (no RGB), which is what we verify here.

NOTE: this is a smoke test, not a full e2e correctness check — that
belongs to Phase 6's ``tests/test_demo_e2e.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo_module():
    spec = importlib.util.spec_from_file_location(
        "_its_demo", REPO_ROOT / "examples" / "its_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_its_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_forest_las(tmp_path: Path) -> Path:
    """Build a small LAS with 3-5 fake trees so the demo has something to
    segment. point_format=1 carries gps_time but no RGB."""
    rng = np.random.default_rng(0)
    tree_centers = np.array(
        [[5.0, 5.0], [15.0, 5.0], [10.0, 15.0]], dtype=np.float64
    )
    points = []
    for cx, cy in tree_centers:
        for _ in range(120):
            r = rng.uniform(0, 3.0)
            theta = rng.uniform(0, 2 * np.pi)
            dx, dy = r * np.cos(theta), r * np.sin(theta)
            h = 12.0 - 1.2 * r + rng.normal(0, 0.4)
            points.append((cx + dx, cy + dy, max(h, 0.1)))
    # ground
    for _ in range(80):
        gx, gy = rng.uniform(0, 20), rng.uniform(0, 20)
        points.append((gx, gy, rng.normal(0, 0.05)))
    arr = np.asarray(points)

    hdr = laspy.LasHeader(point_format=1, version="1.2")
    hdr.scales = (0.001, 0.001, 0.001)
    hdr.offsets = (0.0, 0.0, 0.0)
    las = laspy.LasData(hdr)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.gps_time = np.arange(arr.shape[0], dtype=np.float64) * 1e-3 + 100.0
    las.return_number = np.ones(arr.shape[0], dtype=np.uint8)
    las.number_of_returns = np.ones(arr.shape[0], dtype=np.uint8)
    las.intensity = np.zeros(arr.shape[0], dtype=np.uint16)
    las.classification = np.full(arr.shape[0], 5, dtype=np.uint8)

    p = tmp_path / "synthetic_forest_pf1.las"
    las.write(str(p))
    return p


def test_demo_runs_end_to_end_on_point_format_1(demo_module, tmp_path):
    in_path = _make_synthetic_forest_las(tmp_path)
    out_dir = tmp_path / "demo_out"
    demo_module.main(in_path, out_dir)

    for name in ("li2012.las", "silva2016.las", "dalponte2016.las", "watershed.las"):
        out_path = out_dir / name
        assert out_path.exists(), f"demo failed to write {name}"
        rd = laspy.read(str(out_path))
        # treeID extra dim present
        assert "treeID" in rd.point_format.dimension_names
        # No RGB upgrade — point_format stays 1
        assert rd.point_format.id == 1
