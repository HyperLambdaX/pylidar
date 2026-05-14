"""Raster → point label transfer (lidR ``merge_spatial`` port).

PORT NOTE
---------
Adapted from lidR ``R/segment_trees.R::segment_trees.LAS`` (L23-35) and the
``RasterBased`` branch's call into ``merge_spatial(las, raster, attribute)``
(``R/merge_spatial.R:53-89``), which delegates the raster→point lookup to
``raster_value_from_xy`` → ``raster_cell_from_xy`` (``R/utils_raster.R:55-92``).
The row/col reverse-mapping in :func:`merge_raster_labels` is identical to
:meth:`RasterLayout.cell_xy_to_rowcol` (already 1:1 with
``utils_raster.R:67-73``).

lidR returns ``NA_integer_`` for out-of-grid points (``raster_cell_from_xy``
L69, L73). When the LAS extra-byte is written via
``add_lasattribute_manual(NA_value=.Machine$integer.max)``, that NA propagates
to ``2147483647`` on disk. Phase 5 audit fix #3 (2026-05-12): the default
``nodata`` is now ``None`` and resolves to the LAS-spec NA sentinel that
matches ``labels.dtype`` (``np.iinfo(dtype).max`` for int dtypes,
``np.finfo(dtype).tiny`` for float dtypes), keeping out-of-grid points
consistent with the ``ExtraBytesVlr`` ``no_data`` field written by
:func:`pylidar.io.write_las_with_treeid`. Callers can still pass a
literal ``nodata=0`` to opt into pylidar's in-grid "0 = no tree"
convention for both in- and out-of-grid points.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .raster import RasterLayout

__all__ = ["merge_raster_labels"]


def _default_nodata_for_dtype(dtype: np.dtype) -> Union[int, float]:
    """Return the LAS-spec NA sentinel that matches ``dtype``."""
    if np.issubdtype(dtype, np.integer):
        return int(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.floating):
        return float(np.finfo(dtype).tiny)
    raise TypeError(
        f"merge_raster_labels: cannot derive default nodata for dtype {dtype!r}"
    )


def merge_raster_labels(
    xy: NDArray[np.float64],
    layout: RasterLayout,
    labels: NDArray,
    *,
    nodata: Optional[Union[int, float]] = None,
) -> NDArray:
    """Look up raster ``labels[row, col]`` at each point's (x, y) cell.

    Parameters
    ----------
    xy : (N, 2) float64
        Per-point world coordinates. Same convention as :class:`RasterLayout`
        (X = east-positive, Y = north-positive).
    layout : RasterLayout
        Grid the ``labels`` array lives on.
    labels : (nrow, ncol) ndarray
        Label raster — usually an int32 ITS output, but any 2-D dtype is
        accepted; the returned dtype matches ``labels.dtype``.
    nodata : scalar | None, default None
        Value to assign to points whose ``(x, y)`` falls outside the layout
        bbox. When ``None`` (default), resolves to the LAS-spec NA sentinel
        for ``labels.dtype`` (``np.iinfo(dtype).max`` for int dtypes,
        ``np.finfo(dtype).tiny`` for float dtypes). Pass ``nodata=0``
        explicitly to align out-of-grid points with pylidar's in-grid
        ``0 = no tree`` algorithm convention.

    Returns
    -------
    (N,) ndarray of ``labels.dtype``
    """
    if not isinstance(xy, np.ndarray):
        raise TypeError("xy must be a numpy ndarray")
    if xy.dtype != np.float64:
        raise TypeError("xy must be float64")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy ndarray")
    if labels.ndim != 2:
        raise ValueError("labels must be 2-D")
    if labels.shape != layout.shape:
        raise ValueError(
            f"labels.shape {labels.shape} != layout.shape {layout.shape}"
        )

    resolved_nodata = (
        _default_nodata_for_dtype(labels.dtype) if nodata is None else nodata
    )

    n = xy.shape[0]
    out = np.full(n, resolved_nodata, dtype=labels.dtype)
    if n == 0:
        return out

    row, col = layout.cell_xy_to_rowcol(xy[:, 0], xy[:, 1])
    in_grid = (
        (row >= 0) & (row < layout.nrow) & (col >= 0) & (col < layout.ncol)
    )
    if in_grid.any():
        out[in_grid] = labels[row[in_grid], col[in_grid]]
    return out
