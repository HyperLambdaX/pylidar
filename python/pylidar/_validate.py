"""Input validators shared by the Python wrappers in :mod:`pylidar.segmentation`.

Validation philosophy (spec §6.4 / §8.3): catch user mistakes here in Python
with native ``TypeError``/``ValueError`` so users get a Python-style traceback
that points at *their* call, not deep inside the binding layer. The C++ core
only checks unrecoverable internal invariants.
"""

from __future__ import annotations

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
    """Validate a CHM input. Phase 2+ (lmf / dalponte / silva)."""
    raise NotImplementedError("Phase 2+")


def ensure_transform(transform: object) -> Tuple[float, float, float]:
    """Validate a CHM transform 3-tuple. Phase 2+ (raster algorithms)."""
    raise NotImplementedError("Phase 2+")
