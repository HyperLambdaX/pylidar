"""pylidar — Python port of lidR's individual tree segmentation algorithms.

Top-level convenience re-exports plus the lidR-shaped :func:`segment_trees`
dispatch entry. Modules:

* :mod:`pylidar.io` — LAS/LAZ read + filter helpers, write_las_with_treeid
* :mod:`pylidar.raster` — RasterLayout + rasterize_canopy_* family
* :mod:`pylidar.decimate` — top-of-canopy decimators
* :mod:`pylidar.locate_trees` — Treetops dataclass + locate_trees_chm/_points
* :mod:`pylidar.segmentation` — ITS algorithms (low-level primitives + lidR-shaped wrappers)
* :mod:`pylidar.merge` — raster→point label transfer
* :mod:`pylidar.catalog` — LAScatalog for tile-based wall-to-wall workflows
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from . import catalog, decimate, io, locate_trees, merge, raster, segmentation
from .merge import merge_raster_labels
from .raster import RasterLayout

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "catalog",
    "decimate",
    "io",
    "locate_trees",
    "merge",
    "raster",
    "segmentation",
    "segment_trees",
    "merge_raster_labels",
]


def segment_trees(
    xyz: NDArray[np.float64],
    layout: Optional[RasterLayout] = None,
    *,
    raster_labels: Optional[NDArray] = None,
    point_labels: Optional[NDArray] = None,
    nodata: Optional[float | int] = None,
) -> NDArray:
    """Assign a tree-ID to each input point.

    Mirrors lidR :func:`segment_trees` (``R/segment_trees.R:23-35``): the
    raster branch reverse-maps ``labels[row, col]`` at each point, the point
    branch attaches labels directly.

    Exactly one of ``raster_labels`` / ``point_labels`` must be supplied:

    * **raster path** — ``raster_labels`` is a 2-D label array on ``layout``;
      ``layout`` is required. Out-of-grid points get ``nodata`` (default
      ``None`` → dtype-aware LAS-spec NA sentinel; pass ``nodata=0`` to
      keep out-of-grid points aligned with the in-grid 0 = no-tree
      convention).
    * **point path** — ``point_labels`` is a 1-D length-N array of labels
      already in point order (e.g. ``li2012`` output). ``layout`` ignored.

    Returns
    -------
    (N,) ndarray
        Label per input point. Dtype matches the supplied label array.
    """
    if (raster_labels is None) == (point_labels is None):
        raise ValueError(
            "segment_trees: supply exactly one of raster_labels= or point_labels="
        )
    if not isinstance(xyz, np.ndarray):
        raise TypeError("xyz must be a numpy ndarray")
    if xyz.dtype != np.float64:
        raise TypeError("xyz must be float64")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")

    if raster_labels is not None:
        if layout is None:
            raise ValueError("segment_trees: raster path requires layout=")
        return merge_raster_labels(
            xyz[:, :2], layout, raster_labels, nodata=nodata
        )

    # Point path
    if not isinstance(point_labels, np.ndarray):
        raise TypeError("point_labels must be a numpy ndarray")
    if point_labels.ndim != 1:
        raise ValueError("point_labels must be 1-D")
    if point_labels.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"point_labels length {point_labels.shape[0]} != xyz length {xyz.shape[0]}"
        )
    return point_labels.copy()
