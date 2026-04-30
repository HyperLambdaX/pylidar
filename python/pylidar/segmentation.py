"""High-level Python wrappers around :mod:`pylidar._core`.

Phase 1: ``smooth_height`` — point-cloud Z-value smoothing. Mirrors lidR's
``smooth_height(las, size, method, shape, sigma)``.

Phase 2: ``locate_trees_lmf_chm`` / ``locate_trees_lmf_points`` — local
maximum filter tree-top detection on a CHM raster or an unstructured XYZ
point cloud.

Later phases:
  - ``segment_dalponte2016`` (Phase 3)
  - ``segment_silva2016`` (Phase 4)
  - ``segment_li2012`` (Phase 5)
  - ``segment_watershed`` (Phase 6, pure-Python on ``skimage``)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from . import _core
from ._validate import ensure_chm_float64, ensure_transform, ensure_xyz_float64

__all__ = [
    "locate_trees_lmf_chm",
    "locate_trees_lmf_points",
    "smooth_height",
]


# Map public string args to the small ints the C++ binding takes. "average" is
# accepted as a lidR-compatibility alias for "mean".
_METHOD_MAP = {"mean": 1, "average": 1, "gaussian": 2}
_SHAPE_MAP  = {"square": 1, "circular": 2}


def _resolve_shape(shape: object) -> int:
    """Map a public shape string to the binding's int code."""
    key = shape.lower() if isinstance(shape, str) else shape
    if key not in _SHAPE_MAP:
        raise ValueError(
            f'shape must be one of "circular"/"square", got {shape!r}'
        )
    return _SHAPE_MAP[key]


def _check_ws_hmin(ws: float, hmin: float) -> Tuple[float, float]:
    ws = float(ws)
    if not math.isfinite(ws) or ws <= 0.0:
        raise ValueError(f"ws must be a finite positive number, got {ws}")
    hmin = float(hmin)
    if not math.isfinite(hmin):
        raise ValueError(f"hmin must be a finite number, got {hmin}")
    return ws, hmin


def smooth_height(
    xyz: np.ndarray,
    size: float,
    method: str = "mean",
    shape: str = "circular",
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Smooth the Z values of a point cloud by averaging/Gaussian-weighting
    each point against its 2D-XY neighbours within a ``size``-wide window.

    Ported from lidR ``LAS::z_smooth`` (``src/LAS.cpp:112``).

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) float64, C-contiguous. Columns = (x, y, z).
    size : float
        Window edge length / diameter, world units. Must be > 0.
    method : {"mean", "gaussian"}
        ``"mean"`` is the simple in-window average; ``"gaussian"`` weights
        neighbours by a 2D isotropic Gaussian. ``"average"`` is accepted as a
        lidR-compatibility alias for ``"mean"``.
    shape : {"circular", "square"}
        Neighbourhood shape.
    sigma : float, optional
        Gaussian σ (world units). Required only for ``method="gaussian"``;
        defaults to ``size / 6`` (matches lidR). Must be > 0 when used.

    Returns
    -------
    np.ndarray
        (N,) float64 array of smoothed Z values, in the same order as ``xyz``.

    Raises
    ------
    TypeError
        If ``xyz`` is not a numpy array, or its dtype is not float64.
    ValueError
        On shape ≠ (N, 3), empty input, non-C-contiguous storage, ``size <= 0``,
        unknown method/shape string, or ``sigma <= 0`` when method="gaussian".
    """
    xyz = ensure_xyz_float64(xyz)

    size = float(size)
    # `<= 0` alone lets NaN through silently (NaN <= 0 is False); the C++
    # algorithm would then run a NaN-radius radiusSearch, find no matches,
    # and return the original z values with no error. Reject NaN/inf here.
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError(f"size must be a finite positive number, got {size}")

    method_key = method.lower() if isinstance(method, str) else method
    if method_key not in _METHOD_MAP:
        raise ValueError(
            f'method must be one of "mean"/"average"/"gaussian", got {method!r}'
        )
    method_int = _METHOD_MAP[method_key]

    shape_key = shape.lower() if isinstance(shape, str) else shape
    if shape_key not in _SHAPE_MAP:
        raise ValueError(
            f'shape must be one of "circular"/"square", got {shape!r}'
        )
    shape_int = _SHAPE_MAP[shape_key]

    if sigma is None:
        # lidR default: smooth_height.R defaults sigma = size/6 (only used by
        # the Gaussian branch; Mean ignores it).
        sigma = size / 6.0
    sigma = float(sigma)
    if method_int == 2 and (not math.isfinite(sigma) or sigma <= 0.0):
        raise ValueError(
            f'sigma must be a finite positive number when method="gaussian", '
            f'got {sigma}'
        )

    return _core.smooth_height(xyz, size, method_int, shape_int, sigma)


def locate_trees_lmf_points(
    xyz: np.ndarray,
    ws: float,
    hmin: float = 2.0,
    shape: str = "circular",
) -> np.ndarray:
    """Detect tree tops on an unstructured point cloud via local maximum
    filter.

    Ported from lidR ``LAS::filter_local_maxima(ws, min_height, circular)``
    (``src/LAS.cpp:399``), the variant that ``C_lmf`` calls. Each point is
    a tree-top candidate if it is the highest within a ``ws``-wide
    XY-neighbourhood and its z is at least ``hmin``.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) float64, C-contiguous. Columns = (x, y, z).
    ws : float
        Window edge length / diameter, world units. Must be > 0.
    hmin : float, default 2.0
        Minimum height for a point to be considered a tree top. Points with
        ``z < hmin`` (and points whose z is NaN) are skipped.
    shape : {"circular", "square"}, default "circular"
        Neighbourhood shape.

    Returns
    -------
    np.ndarray
        (M, 3) float64 array of detected tree tops; columns = (x, y, z).
        ``M == 0`` is returned as an empty ``(0, 3)`` array. The output
        carries no IDs — downstream segmentation algorithms (Phase 3+)
        assign 1..M as needed.
    """
    xyz = ensure_xyz_float64(xyz)
    ws, hmin = _check_ws_hmin(ws, hmin)
    shape_int = _resolve_shape(shape)
    return _core.lmf_points(xyz, ws, hmin, shape_int)


def locate_trees_lmf_chm(
    chm: np.ndarray,
    transform: Tuple[float, float, float],
    ws: float,
    hmin: float = 2.0,
    shape: str = "circular",
) -> np.ndarray:
    """Detect tree tops on a CHM raster via local maximum filter.

    Mirrors lidR's raster path: ``locate_trees(<SpatRaster>, lmf(...))``
    (R/locate_trees.R:136-148) rasterises the CHM into a fake
    1-point-per-cell LAS via ``raster_as_las()`` and runs the same
    point-cloud LMF. We do the equivalent — non-NaN cells become a virtual
    point cloud (XY = pixel-centre world coords) and the result is
    returned as XYZ tree tops.

    Parameters
    ----------
    chm : np.ndarray
        (H, W) float64, C-contiguous, row-major. NaN cells are masked.
    transform : tuple of (origin_x, origin_y, pixel_size)
        Spec §6.1: ``origin_*`` is the world XY of the pixel ``(row=0,
        col=0)`` *centre*; row 0 is the northern edge (largest y);
        ``pixel_size > 0`` is an isotropic edge length. No rotation/shear.
    ws : float
        Window edge length / diameter, **world** units (not pixels).
    hmin : float, default 2.0
        Minimum pixel value to be considered a tree top.
    shape : {"circular", "square"}, default "circular"
        Neighbourhood shape.

    Returns
    -------
    np.ndarray
        (M, 3) float64 array of detected tree tops; columns =
        (x_world, y_world, z) where z is the CHM pixel value.
    """
    chm = ensure_chm_float64(chm)
    ox, oy, ps = ensure_transform(transform)
    ws, hmin = _check_ws_hmin(ws, hmin)
    shape_int = _resolve_shape(shape)
    return _core.lmf_chm(chm, ox, oy, ps, ws, hmin, shape_int)
