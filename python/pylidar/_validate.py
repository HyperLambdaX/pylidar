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
    "ensure_seeds_xyzid",
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


def ensure_seeds_xyzid(arr: object, *, name: str = "seeds") -> np.ndarray:
    """Validate / pack a tree-top seeds array for segmentation algorithms.

    Spec §7: callers may pass either ``(M, 3)`` (columns x, y, z, no IDs)
    or ``(M, 4)`` (columns x, y, z, id). The (M, 3) form is auto-IDed
    ``1..M``; the (M, 4) form is used as-is. Returns a fresh
    C-contiguous ``(M, 4)`` float64 array. ``M == 0`` is allowed and
    short-circuits to an empty (0, 4) array — useful for batch flows
    where no trees were detected.

    XY/Z columns must be finite (matches lidR, which errors at
    ``sf::st_rasterize`` on NaN/Inf point coordinates — without this
    check the C++ algorithm would silently drop them, divergent from
    upstream). IDs in the (M, 4) form must be ≥ 1 (spec §3 / §7: the
    output raster uses ``0`` as the "no tree" sentinel and negative
    labels would silently break downstream ``mask = crowns > 0``
    patterns).

    - ``TypeError`` on non-ndarray or non-float64 dtype.
    - ``ValueError`` on shape ≠ (M, 3) / (M, 4), non-finite x/y/z, or
      ID < 1 in the (M, 4) form.
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
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError(
            f"{name} must have shape (M, 3) or (M, 4), got {arr.shape}"
        )

    m = arr.shape[0]
    if m == 0:
        return np.empty((0, 4), dtype=np.float64)

    # XY/Z must be finite (issue #4: lidR errors on NaN/Inf coords; we
    # used to silently drop them in C++, now we reject up-front).
    if not np.all(np.isfinite(arr[:, :3])):
        raise ValueError(
            f"{name} x/y/z values must all be finite (no NaN or Inf)"
        )

    if arr.shape[1] == 3:
        out = np.empty((m, 4), dtype=np.float64)
        out[:, :3] = arr
        out[:, 3]  = np.arange(1, m + 1, dtype=np.float64)
        return out

    # (M, 4) branch — validate the user-supplied IDs.
    ids = arr[:, 3]
    if not np.all(np.isfinite(ids)):
        raise ValueError(
            f"{name} IDs (column 3) must all be finite numbers"
        )
    # Spec §3 / §7: tree IDs are ≥ 1 (0 = "no tree" sentinel; negative
    # values would silently bypass downstream `crowns > 0` masks).
    ids_int = ids.astype(np.int32)
    if np.any(ids_int < 1):
        raise ValueError(
            f"{name} IDs (column 3) must all be >= 1 "
            f"(0 is reserved for 'no tree'; negative values are not allowed)"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


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
