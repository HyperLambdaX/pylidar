"""Point-cloud decimation primitives (lidR ``decimate_points`` family port).

PORT NOTE
---------
Adapted from lidR ``R/algorithm-dec.R`` (L157-173 ``highest``) and
``src/LAS.cpp`` (L605-617 the C kernel). The strict ``<`` tie-break
(``if (zref < z) keep = i;``) keeps the first-encountered point on equal-Z
ties — matches the C source exactly.

This replaces the deprecated lidR ``filter_surfacepoints``
(``R/deprecated.R:52-62``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .raster import RasterLayout

__all__ = ["decimate_highest"]


def decimate_highest(
    xyz: NDArray[np.float64],
    layout: RasterLayout,
) -> NDArray[np.bool_]:
    """Per-cell argmax over Z; ties keep the first-encountered point.

    Returns a length-N boolean mask: ``True`` where the point survives.
    Points falling outside ``layout`` are dropped (``False``).
    """
    if not isinstance(xyz, np.ndarray):
        raise TypeError("decimate_highest: xyz must be a numpy ndarray")
    if xyz.dtype != np.float64:
        raise TypeError("decimate_highest: xyz must be float64")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("decimate_highest: xyz must have shape (N, 3)")

    n = xyz.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return keep

    row, col = layout.cell_xy_to_rowcol(xyz[:, 0], xyz[:, 1])
    in_grid = (row >= 0) & (row < layout.nrow) & (col >= 0) & (col < layout.ncol)

    # Per-cell argmax with strict-< tie-break (first-encountered wins).
    # Walk in order: for each point in `in_grid`, only replace cell's
    # current best if z strictly greater.
    flat_cell = row * layout.ncol + col
    best_idx = np.full(layout.nrow * layout.ncol, -1, dtype=np.int64)
    best_z = np.full(layout.nrow * layout.ncol, -np.inf, dtype=np.float64)

    z = xyz[:, 2]
    # Vectorize impossible (sequential dependence on best_z); loop in Python.
    for i in range(n):
        if not in_grid[i]:
            continue
        c = flat_cell[i]
        if best_z[c] < z[i]:  # strict < per src/LAS.cpp:613
            best_idx[c] = i
            best_z[c] = z[i]

    surviving = best_idx[best_idx >= 0]
    keep[surviving] = True
    return keep
