"""Input validators shared by the Python wrappers in :mod:`pylidar.segmentation`.

Validation philosophy (spec §6.4 / §8.3): catch user mistakes here in Python
with native ``TypeError``/``ValueError`` so users get a Python-style traceback
that points at *their* call, not deep inside the binding layer. The C++ core
only checks unrecoverable internal invariants.

Phase 0: this module only declares the function surface — implementations land
in Phase 1 alongside the first algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "ensure_chm_float64",
    "ensure_xyz_float64",
    "ensure_transform",
]


def ensure_chm_float64(arr: "np.ndarray", *, name: str = "chm") -> "np.ndarray":
    """Validate a CHM input. Returns ``arr`` unchanged on success.

    Raises ``TypeError`` if dtype isn't float64. Raises ``ValueError`` if
    ``arr.ndim != 2`` or the array isn't C-contiguous.
    """
    raise NotImplementedError("Phase 1+")


def ensure_xyz_float64(arr: "np.ndarray", *, name: str = "xyz") -> "np.ndarray":
    """Validate an (N, 3) point cloud. Returns ``arr`` unchanged on success.

    Raises ``TypeError`` if dtype isn't float64. Raises ``ValueError`` on
    shape != (N, 3) or non-C-contiguous storage.
    """
    raise NotImplementedError("Phase 1+")


def ensure_transform(transform: object) -> Tuple[float, float, float]:
    """Validate a CHM transform 3-tuple ``(origin_x, origin_y, pixel_size)``.

    Raises ``TypeError`` if not a 3-tuple of floats. Raises ``ValueError`` if
    ``pixel_size <= 0``.
    """
    raise NotImplementedError("Phase 1+")
