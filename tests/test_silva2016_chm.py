"""silva2016_chm — CHM-cell Voronoi + per-tree hmax filter.

1:1 with lidR ``R/algorithm-its.R::silva2016`` (L257-277). Differs from
the existing point-cloud :func:`pylidar.segmentation.silva2016` in that
``hmax`` is computed over CHM cells (not point z values).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylidar.locate_trees import Treetops
from pylidar.raster import RasterLayout
from pylidar.segmentation import silva2016_chm


def _layout(h, w):
    return RasterLayout(
        xmin=0.0, ymax=float(h), xres=1.0, yres=1.0, ncol=w, nrow=h, crs=None,
    )


def test_silva_chm_two_trees_voronoi_split():
    """A 5x10 CHM with two equal-height trees splits along the midline."""
    chm = np.full((5, 10), 6.0, dtype=np.float64)
    layout = _layout(5, 10)
    # Treetop 1 at xy = (2.5, 2.5) → row=2, col=2; treetop 2 at (7.5, 2.5)
    x = np.array([2.5, 7.5], dtype=np.float64)
    y = np.array([2.5, 2.5], dtype=np.float64)
    z = np.array([6.0, 6.0], dtype=np.float64)
    tid = np.array([1, 2], dtype=np.int32)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    out = silva2016_chm(chm=chm, layout=layout, treetops=tt)
    # All cells assigned (height 6.0 > exclusion=0.3*6 = 1.8 and within
    # max_cr_factor=0.6*6 = 3.6 of one of the two treetops, modulo corners).
    # Verify Voronoi split: leftmost cells → tree 1, rightmost → tree 2.
    assert out[2, 0] == 1  # closer to (2.5, 2.5)
    assert out[2, 9] == 2  # closer to (7.5, 2.5)


def test_silva_chm_exclusion_drops_low_cells():
    """Cells below exclusion*hmax become 0."""
    chm = np.array([
        [9, 9, 9, 9],
        [9, 9, 9, 9],
        [1, 1, 1, 1],   # row of low cells (z=1 < 0.3*9 = 2.7)
        [9, 9, 9, 9],
    ], dtype=np.float64)
    layout = _layout(4, 4)
    x = np.array([2.0], dtype=np.float64)
    y = np.array([3.5], dtype=np.float64)  # row=0
    z = np.array([9.0], dtype=np.float64)
    tid = np.array([1], dtype=np.int32)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    out = silva2016_chm(
        chm=chm, layout=layout, treetops=tt,
        exclusion=0.3, max_cr_factor=10.0,  # large factor so distance never bites
    )
    # Low row stays 0; rest becomes 1.
    assert (out[2, :] == 0).all()
    # At least the high rows are assigned; corner cells may exceed
    # max_cr_factor but factor=10 is huge, so they should be in.
    assert (out[0, :] == 1).all()
    assert (out[3, :] == 1).all()


def test_silva_chm_distance_filter_caps_crown_radius():
    """Cells beyond max_cr_factor*hmax of treetop become 0."""
    chm = np.full((20, 20), 5.0, dtype=np.float64)
    layout = _layout(20, 20)
    x = np.array([10.0], dtype=np.float64)
    y = np.array([10.0], dtype=np.float64)
    z = np.array([5.0], dtype=np.float64)
    tid = np.array([1], dtype=np.int32)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    out = silva2016_chm(
        chm=chm, layout=layout, treetops=tt,
        exclusion=0.01,         # height filter inactive
        max_cr_factor=0.6,      # crown radius = 0.6 * 5 = 3.0
    )
    # Cells at distance > 3.0 from (10, 10) must be 0.
    rows, cols = np.indices(chm.shape)
    cell_x = (cols + 0.5) * 1.0  # layout.xres=1, xmin=0
    cell_y = 20.0 - (rows + 0.5) * 1.0
    d = np.sqrt((cell_x - 10.0) ** 2 + (cell_y - 10.0) ** 2)
    far = d > 3.0
    assert (out[far] == 0).all()
    # Some near cells must be assigned.
    assert (out[~far] == 1).any()


def test_silva_chm_chm_with_nans_skipped():
    chm = np.full((5, 5), 6.0, dtype=np.float64)
    chm[0, 0] = np.nan
    layout = _layout(5, 5)
    x = np.array([2.5], dtype=np.float64)
    y = np.array([2.5], dtype=np.float64)
    z = np.array([6.0], dtype=np.float64)
    tid = np.array([1], dtype=np.int32)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    out = silva2016_chm(
        chm=chm, layout=layout, treetops=tt, max_cr_factor=10.0,
    )
    # NaN cell stays 0 in output.
    assert out[0, 0] == 0


def test_silva_chm_empty_treetops_returns_zeros():
    chm = np.full((5, 5), 6.0, dtype=np.float64)
    layout = _layout(5, 5)
    e = np.empty((0,), dtype=np.float64)
    eid = np.empty((0,), dtype=np.int32)
    tt = Treetops(x=e, y=e, z=e, tree_id=eid)
    out = silva2016_chm(chm=chm, layout=layout, treetops=tt)
    assert out.shape == (5, 5)
    assert (out == 0).all()


def test_silva_chm_propagates_float64_ids():
    chm = np.full((5, 5), 6.0, dtype=np.float64)
    layout = _layout(5, 5)
    x = np.array([2.5], dtype=np.float64)
    y = np.array([2.5], dtype=np.float64)
    z = np.array([6.0], dtype=np.float64)
    tid = np.array([300123.456], dtype=np.float64)
    tt = Treetops(x=x, y=y, z=z, tree_id=tid)
    out = silva2016_chm(
        chm=chm, layout=layout, treetops=tt, max_cr_factor=10.0,
    )
    assert out.dtype == np.float64
    nz = np.unique(out[out != 0.0])
    assert nz.tolist() == [300123.456]


def test_silva_chm_rejects_non_treetops():
    chm = np.full((5, 5), 6.0, dtype=np.float64)
    layout = _layout(5, 5)
    with pytest.raises(TypeError, match="Treetops"):
        silva2016_chm(chm=chm, layout=layout, treetops="not a Treetops")


def test_silva_chm_rejects_invalid_exclusion():
    chm = np.full((5, 5), 6.0, dtype=np.float64)
    layout = _layout(5, 5)
    e = np.empty((0,), dtype=np.float64)
    eid = np.empty((0,), dtype=np.int32)
    tt = Treetops(x=e, y=e, z=e, tree_id=eid)
    with pytest.raises(ValueError, match="exclusion"):
        silva2016_chm(chm=chm, layout=layout, treetops=tt, exclusion=1.5)
