"""dalponte2016_from_treetops — high-level seeded region growing.

Mirrors lidR ``R/algorithm-its.R::dalponte2016`` (L118-140): seed grid
built from Treetops, _core.dalponte2016 runs, sequential IDs remap back
to ``treetops.tree_id``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.locate_trees import Treetops, locate_trees_chm
from pylidar.raster import RasterLayout
from pylidar.segmentation import dalponte2016_from_treetops


def _two_peak_chm():
    """7x7 CHM with two clearly separated peaks at (1,1) and (5,5)."""
    chm = np.zeros((7, 7), dtype=np.float64)
    # Peak 1 at row=1, col=1, height 8
    chm[1, 1] = 8.0
    chm[0:3, 0:3] = np.maximum(chm[0:3, 0:3], 4.0)  # crown
    chm[1, 1] = 8.0
    # Peak 2 at row=5, col=5, height 9
    chm[5, 5] = 9.0
    chm[4:7, 4:7] = np.maximum(chm[4:7, 4:7], 5.0)
    chm[5, 5] = 9.0
    return chm


def _layout(h=7, w=7):
    return RasterLayout(
        xmin=0.0, ymax=float(h), xres=1.0, yres=1.0, ncol=w, nrow=h, crs=None,
    )


def test_dalponte_from_treetops_happy_two_peaks():
    chm = _two_peak_chm()
    layout = _layout()
    tt = locate_trees_chm(chm, layout, ws=3.0, hmin=2.0)
    assert tt.n == 2
    out = dalponte2016_from_treetops(chm=chm, layout=layout, treetops=tt)
    assert out.shape == (7, 7)
    assert out.dtype == np.int32
    # Two distinct positive labels
    labels = set(int(v) for v in np.unique(out) if v > 0)
    assert labels == {1, 2}
    # The two seed cells must hold their own labels.
    rows, cols = layout.cell_xy_to_rowcol(tt.x, tt.y)
    assert out[rows[0], cols[0]] == tt.tree_id[0]
    assert out[rows[1], cols[1]] == tt.tree_id[1]


def test_dalponte_from_treetops_remaps_to_user_ids_float64():
    """User-provided gpstime/bitmerge IDs (float64) propagate through."""
    chm = _two_peak_chm()
    layout = _layout()
    # Manually build a Treetops with float64 ids to exercise the remap.
    rows, cols = np.array([1, 5], dtype=np.int64), np.array([1, 5], dtype=np.int64)
    x, y = layout.rowcol_to_cell_xy(rows, cols)
    z = chm[rows, cols].astype(np.float64)
    custom_ids = np.array([300123.456, 300456.789], dtype=np.float64)
    tt = Treetops(x=x, y=y, z=z, tree_id=custom_ids)
    out = dalponte2016_from_treetops(chm=chm, layout=layout, treetops=tt)
    assert out.dtype == np.float64
    # Both custom IDs present in the output, plus 0.0 for unassigned.
    nz = np.unique(out[out != 0.0])
    assert set(nz.tolist()) == set(custom_ids.tolist())


def test_dalponte_from_treetops_empty_treetops_returns_zeros():
    chm = _two_peak_chm()
    layout = _layout()
    e = np.empty((0,), dtype=np.float64)
    eid = np.empty((0,), dtype=np.int32)
    tt = Treetops(x=e, y=e, z=e, tree_id=eid)
    out = dalponte2016_from_treetops(chm=chm, layout=layout, treetops=tt)
    assert out.shape == (7, 7)
    assert out.dtype == np.int32
    assert (out == 0).all()


def test_dalponte_from_treetops_treetops_outside_layout_skipped():
    """Treetops whose xy fall outside the layout extent are not seeded."""
    chm = _two_peak_chm()
    layout = _layout()
    # Two seeds: one inside (will seed cell (1,1)), one wildly outside.
    x = np.array([1.5, 100.0], dtype=np.float64)
    y = np.array([5.5, 100.0], dtype=np.float64)  # row=1.5 in our layout
    z = np.array([8.0, 0.0], dtype=np.float64)
    tid = np.array([1, 2], dtype=np.int32)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    out = dalponte2016_from_treetops(chm=chm, layout=layout, treetops=tt)
    # Only label 1 should appear (the in-bounds seed).
    nz = set(int(v) for v in np.unique(out) if v > 0)
    assert nz == {1}


def test_dalponte_from_treetops_chm_with_nans():
    """NaN cells in CHM are treated as -inf — they don't grow into trees."""
    chm = _two_peak_chm()
    chm[3, 3] = np.nan  # center cell — must stay 0 in output
    layout = _layout()
    tt = locate_trees_chm(chm, layout, ws=3.0, hmin=2.0)
    out = dalponte2016_from_treetops(chm=chm, layout=layout, treetops=tt)
    assert out[3, 3] == 0


def test_dalponte_from_treetops_rejects_non_treetops():
    chm = _two_peak_chm()
    layout = _layout()
    with pytest.raises(TypeError, match="Treetops"):
        dalponte2016_from_treetops(chm=chm, layout=layout, treetops="not a Treetops")


def test_dalponte_from_treetops_rejects_chm_layout_mismatch():
    chm = _two_peak_chm()
    layout = _layout(h=10)  # wrong nrow
    e = np.empty((0,), dtype=np.float64)
    eid = np.empty((0,), dtype=np.int32)
    tt = Treetops(x=e, y=e, z=e, tree_id=eid)
    with pytest.raises(ValueError, match="match layout.shape"):
        dalponte2016_from_treetops(chm=chm, layout=layout, treetops=tt)


def test_dalponte_from_treetops_collision_last_wins():
    """Two treetops in the same cell — lidR semantics: last write wins."""
    chm = np.zeros((5, 5), dtype=np.float64)
    chm[2, 2] = 9.0
    chm[1:4, 1:4] = np.maximum(chm[1:4, 1:4], 4.0)
    chm[2, 2] = 9.0
    layout = _layout(5, 5)
    # Both treetops fall in cell (2, 2) of this layout.
    x = np.array([2.5, 2.4], dtype=np.float64)
    y = np.array([2.5, 2.6], dtype=np.float64)
    z = np.array([9.0, 9.0], dtype=np.float64)
    tid = np.array([10, 20], dtype=np.int32)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    out = dalponte2016_from_treetops(chm=chm, layout=layout, treetops=tt)
    # Last write wins → seed cell becomes 2 (sequential index of second tree),
    # which remaps to tree_id[1] = 20.
    nz = set(int(v) for v in np.unique(out) if v > 0)
    assert nz == {20}
