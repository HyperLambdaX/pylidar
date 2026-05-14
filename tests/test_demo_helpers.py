"""Targeted tests for the examples/its_demo.py helper that gates RGB on the
input LAS point format.

Phase 5 audit-fix #2 (2026-05-12): the original demo always passed ``rgb=``
to ``write_las_with_treeid``, which raises on non-RGB formats (0/1/4/6/9).
The helper now returns ``None`` for those formats so the demo runs
end-to-end with treeID-only output.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo_module():
    """Import examples/its_demo.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "_its_demo", REPO_ROOT / "examples" / "its_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_its_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("pf", [0, 1])
def test_rgb_helper_returns_none_for_non_rgb_format(
    demo_module, build_synthetic_las, pf
):
    """LAS 1.2-compatible non-RGB formats: 0 (basic), 1 (gps_time).
    Formats 4/6/9 require LAS 1.3+/1.4+; the conftest factory pins 1.2."""
    las = build_synthetic_las(point_format=pf)
    n = len(las.x)
    tree_id = np.arange(1, n + 1, dtype=np.int32)
    rgb = demo_module._rgb_for_writer(tree_id, las)
    assert rgb is None


@pytest.mark.parametrize("pf", [2, 3])
def test_rgb_helper_returns_colours_for_rgb_format(
    demo_module, build_synthetic_las, pf
):
    las = build_synthetic_las(point_format=pf)
    n = len(las.x)
    tree_id = np.arange(1, n + 1, dtype=np.int32)
    rgb = demo_module._rgb_for_writer(tree_id, las)
    assert rgb is not None
    assert rgb.shape == (n, 3)
    assert rgb.dtype == np.uint16


def test_rgb_helper_ignores_int32_na_sentinel(demo_module):
    """The LAS NA sentinel (int32.max) for out-of-grid points must not
    leak into the palette sizing or the colour assignment.

    Pre-fix, ``tree_id.max()`` returned ~2.1 billion (the sentinel) and
    the helper attempted to build a ~50 GB palette. Post-fix the helper
    only counts real labels.
    """
    na = np.iinfo(np.int32).max
    tree_id = np.array([1, 2, 3, na, 0, 2, na], dtype=np.int32)
    rgb = demo_module._rgb_for_tree_id(tree_id, seed=0)
    assert rgb.shape == (7, 3)
    assert rgb.dtype == np.uint16
    # Sentinel rows must be coloured neutral grey, not whatever the
    # palette at index `na - 1` would have yielded.
    grey = int(round(0.4 * 65535))
    for idx in (3, 4, 6):  # NA, no-tree, NA
        assert tuple(int(c) for c in rgb[idx]) == (grey, grey, grey), (
            f"row {idx} (sentinel/no-tree) was coloured, not grey"
        )
    # Real-tree rows must NOT be the neutral grey colour.
    for idx in (0, 1, 2, 5):
        assert tuple(int(c) for c in rgb[idx]) != (grey, grey, grey)


def test_summary_excludes_na_sentinel(demo_module, capsys):
    """`_summary` must not count the NA sentinel as a tree."""
    na = np.iinfo(np.int32).max
    # Four real trees (max label 4), plus a sentinel.
    tree_id = np.array([1, 2, 3, 4, 0, na], dtype=np.int32)
    demo_module._summary("test", 0.0, tree_id)
    out = capsys.readouterr().out
    # Real tree count is 4; pre-fix would have reported 2147483647.
    assert "4 trees" in out, f"expected '4 trees' in summary, got: {out!r}"
    # 4 of 6 are valid → 66.7%, not 100% (pre-fix counted sentinel as valid).
    assert "100.0% assigned" not in out, (
        f"summary mis-counts NA sentinel as assigned: {out!r}"
    )
