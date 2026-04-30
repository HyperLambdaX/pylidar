"""Input validators shared by the Python wrappers in :mod:`pylidar.segmentation`.

Validation philosophy (spec §6.4 / §8.3): catch user mistakes here in Python
with native ``TypeError``/``ValueError`` so users get a Python-style traceback
that points at *their* call, not deep inside the binding layer. The C++ core
only checks unrecoverable internal invariants.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = [
    "ensure_chm_float64",
    "ensure_xyz_float64",
    "ensure_transform",
]


def ensure_xyz_float64(arr: object, *, name: str = "xyz") -> np.ndarray:
    """Validate an ``(N, 3)`` float64 C-contiguous point cloud.

    Returns ``arr`` unchanged on success.

    - ``TypeError`` if ``arr`` is not a numpy array or its dtype isn't
      ``float64``. The message names the offending arg and suggests
      ``.astype(np.float64)``.
    - ``ValueError`` if shape is not ``(N, 3)``, ``N == 0``, or storage is not
      C-contiguous (suggests ``np.ascontiguousarray``).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(arr).__name__}"
        )
    if arr.dtype != np.float64:
        raise TypeError(
            f"{name} must have dtype=float64, got {arr.dtype}. "
            f"Convert with {name}.astype(np.float64) before calling."
        )
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (N, 3), got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point, got N=0")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"{name} must be C-contiguous. "
            f"Wrap with np.ascontiguousarray({name}) before calling."
        )
    return arr


def ensure_chm_float64(arr: object, *, name: str = "chm") -> np.ndarray:
    """Validate a CHM raster: ``(H, W)`` float64 C-contiguous, with H, W ≥ 1.

    Returns ``arr`` unchanged on success.

    - ``TypeError`` if ``arr`` is not a numpy array or its dtype isn't
      ``float64`` (suggests ``.astype(np.float64)``).
    - ``ValueError`` if it isn't 2D, has a zero dimension, or isn't
      C-contiguous (suggests ``np.ascontiguousarray``). NaN cells are
      *allowed* — algorithms treat them as masked.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(arr).__name__}"
        )
    if arr.dtype != np.float64:
        raise TypeError(
            f"{name} must have dtype=float64, got {arr.dtype}. "
            f"Convert with {name}.astype(np.float64) before calling."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must have shape (H, W), got {arr.shape}"
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(
            f"{name} must have non-zero rows and cols, got {arr.shape}"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"{name} must be C-contiguous. "
            f"Wrap with np.ascontiguousarray({name}) before calling."
        )
    return arr


def ensure_transform(transform: object) -> Tuple[float, float, float]:
    """Validate a CHM affine transform tuple ``(origin_x, origin_y, pixel_size)``.

    Returns the same triple coerced to ``float``. Spec §6.1 / §7: this is
    the GIS-style triple where ``origin_*`` is the world coordinate of the
    pixel ``(row=0, col=0)`` *centre* and ``pixel_size > 0`` is an
    isotropic edge length. Rotation/shear are out of scope.
    """
    if not isinstance(transform, (tuple, list)):
        raise TypeError(
            f"transform must be a 3-tuple (origin_x, origin_y, pixel_size), "
            f"got {type(transform).__name__}"
        )
    if len(transform) != 3:
        raise ValueError(
            f"transform must have exactly 3 elements (origin_x, origin_y, "
            f"pixel_size), got len={len(transform)}"
        )
    try:
        ox = float(transform[0])
        oy = float(transform[1])
        ps = float(transform[2])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"transform elements must be numeric, got {transform!r}"
        ) from exc

    if not math.isfinite(ox) or not math.isfinite(oy):
        raise ValueError(
            f"transform origin_x/origin_y must be finite, got "
            f"({ox}, {oy})"
        )
    if not math.isfinite(ps) or ps <= 0.0:
        raise ValueError(
            f"transform pixel_size must be a finite positive number, got {ps}"
        )
    return (ox, oy, ps)
