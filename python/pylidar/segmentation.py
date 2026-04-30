"""High-level Python wrappers around :mod:`pylidar._core`.

Phase 1: ``smooth_height`` — point-cloud Z-value smoothing. Mirrors lidR's
``smooth_height(las, size, method, shape, sigma)``.

Later phases:
  - ``locate_trees_lmf_chm`` / ``locate_trees_lmf_points`` (Phase 2)
  - ``segment_dalponte2016`` (Phase 3)
  - ``segment_silva2016`` (Phase 4)
  - ``segment_li2012`` (Phase 5)
  - ``segment_watershed`` (Phase 6, pure-Python on ``skimage``)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import _core
from ._validate import ensure_xyz_float64

__all__ = ["smooth_height"]


# Map public string args to the small ints the C++ binding takes. "average" is
# accepted as a lidR-compatibility alias for "mean".
_METHOD_MAP = {"mean": 1, "average": 1, "gaussian": 2}
_SHAPE_MAP  = {"square": 1, "circular": 2}


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
