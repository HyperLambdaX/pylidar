"""Height normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._interp import knnidw_interpolate, kriging_interpolate, tin_interpolate

__all__ = [
    "NormalizedHeightResult",
    "normalize_height",
    "normalize_height_with_metadata",
]


NormalizeMethod = Literal["tin", "knnidw", "kriging"]


@dataclass(frozen=True)
class NormalizedHeightResult:
    z: NDArray[np.float64]
    ground_z: NDArray[np.float64]
    method: str
    fallback_counts: dict[str, int]


def _validate_inputs(
    xyz: NDArray[np.float64],
    ground_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    xyz_arr = np.ascontiguousarray(xyz, dtype=np.float64)
    if xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3:
        raise ValueError("normalize_height: xyz must have shape (N, 3)")
    if not np.all(np.isfinite(xyz_arr)):
        raise ValueError("normalize_height: xyz must be finite")
    ground = np.ascontiguousarray(ground_mask, dtype=np.bool_)
    if ground.ndim != 1:
        raise ValueError("normalize_height: ground_mask must be 1-D")
    if ground.shape[0] != xyz_arr.shape[0]:
        raise ValueError("normalize_height: ground_mask length must equal xyz row count")
    if not ground.any():
        raise ValueError("normalize_height: ground_mask contains no ground points")
    return xyz_arr, ground


def _nearest_rescue(
    src_xy: NDArray[np.float64],
    src_z: NDArray[np.float64],
    query_xy: NDArray[np.float64],
    ground_z: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int]:
    # Robustness layer beyond lidR's tin(extrapolate=knnidw(3,1,50)) default:
    # if TIN and bounded KNN-IDW still leave gaps, use nearest ground so
    # callers do not silently carry NaN normalized heights. The count is
    # surfaced in metadata for demos/audits.
    missing = np.isnan(ground_z)
    if not missing.any():
        return ground_z, 0
    rescued = knnidw_interpolate(
        src_xy,
        src_z,
        query_xy[missing],
        k=1,
        p=1.0,
        rmax=np.inf,
    )
    out = ground_z.copy()
    out[missing] = rescued
    return out, int(np.count_nonzero(np.isfinite(rescued)))


def normalize_height_with_metadata(
    *,
    xyz: NDArray[np.float64],
    ground_mask: NDArray[np.bool_],
    method: NormalizeMethod = "tin",
) -> NormalizedHeightResult:
    """Return normalized heights plus interpolation metadata."""
    xyz_arr, ground = _validate_inputs(xyz, ground_mask)
    if method not in ("tin", "knnidw", "kriging"):
        raise ValueError("normalize_height: method must be 'tin', 'knnidw', or 'kriging'")

    src_xy = xyz_arr[ground, :2]
    src_z = xyz_arr[ground, 2]
    query_xy = xyz_arr[:, :2]
    counts = {
        "tin_missing": 0,
        "knnidw_extrapolated": 0,
        "nearest_rescued": 0,
        "unresolved": 0,
    }

    if method == "tin":
        if src_xy.shape[0] < 3:
            raise ValueError("normalize_height: tin requires at least 3 ground points")
        ground_z = tin_interpolate(src_xy, src_z, query_xy, extrapolate=None)
        missing = np.isnan(ground_z)
        counts["tin_missing"] = int(np.count_nonzero(missing))
        if missing.any():
            fill = knnidw_interpolate(
                src_xy,
                src_z,
                query_xy[missing],
                k=3,
                p=1.0,
                rmax=50.0,
            )
            ground_z[missing] = fill
            counts["knnidw_extrapolated"] = int(np.count_nonzero(np.isfinite(fill)))
    elif method == "knnidw":
        ground_z = knnidw_interpolate(src_xy, src_z, query_xy)
    else:
        ground_z = kriging_interpolate(src_xy, src_z, query_xy)

    ground_z, rescued = _nearest_rescue(src_xy, src_z, query_xy, ground_z)
    counts["nearest_rescued"] = rescued
    counts["unresolved"] = int(np.count_nonzero(np.isnan(ground_z)))
    if counts["unresolved"]:
        raise ValueError("normalize_height: interpolation left unresolved points")

    return NormalizedHeightResult(
        z=np.ascontiguousarray(xyz_arr[:, 2] - ground_z, dtype=np.float64),
        ground_z=np.ascontiguousarray(ground_z, dtype=np.float64),
        method=method,
        fallback_counts=counts,
    )


def normalize_height(
    *,
    xyz: NDArray[np.float64],
    ground_mask: NDArray[np.bool_],
    method: NormalizeMethod = "tin",
) -> NDArray[np.float64]:
    """Return height-above-ground values for every input point."""
    return normalize_height_with_metadata(
        xyz=xyz,
        ground_mask=ground_mask,
        method=method,
    ).z
