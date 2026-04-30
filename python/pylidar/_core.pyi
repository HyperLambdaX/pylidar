"""Type stubs for the C++ extension `pylidar._core`.

Hand-written; kept in sync with src/bindings/module.cpp. Algorithm function
stubs are added phase-by-phase as their bindings land.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

def set_log_callback(callback: Optional[Callable[[str], None]]) -> None:
    """Install a Python callable to receive log messages from the C++ core.

    Pass ``None`` to disable logging (the default). Exceptions raised by the
    callback are swallowed and routed through :func:`sys.unraisablehook`.
    """
    ...

def smooth_height(
    xyz: npt.NDArray[np.float64],
    size: float,
    method: int,
    shape: int,
    sigma: float,
) -> npt.NDArray[np.float64]:
    """Internal: smooth point-cloud Z values.

    method: 1=mean, 2=gaussian. shape: 1=square, 2=circular.

    Use :func:`pylidar.smooth_height` instead — it validates inputs and
    accepts the public string-based API.
    """
    ...

def lmf_points(
    xyz: npt.NDArray[np.float64],
    ws: float,
    hmin: float,
    shape: int,
) -> npt.NDArray[np.float64]:
    """Internal: local-maximum-filter tree-top detection on (N, 3) XYZ.

    Returns an (M, 3) float64 array of (x, y, z) tree tops.
    shape: 1=square, 2=circular.

    Use :func:`pylidar.locate_trees_lmf_points` instead.
    """
    ...

def lmf_chm(
    chm: npt.NDArray[np.float64],
    origin_x: float,
    origin_y: float,
    pixel_size: float,
    ws: float,
    hmin: float,
    shape: int,
) -> npt.NDArray[np.float64]:
    """Internal: local-maximum-filter tree-top detection on a CHM raster.

    Caller passes the CHM as a row-major (H, W) float64 array plus an
    unpacked (origin_x, origin_y, pixel_size) triple. Returns (M, 3)
    float64 (x_world, y_world, z).

    Use :func:`pylidar.locate_trees_lmf_chm` instead.
    """
    ...
