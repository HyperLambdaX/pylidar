"""Generic scattered interpolation helpers shared by raster and normalization.

The helpers operate on explicit source/query coordinates instead of raster
layouts. Raster callers keep any raster-specific masking outside this module;
normalization callers can query directly at point XY coordinates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

__all__ = [
    "tin_interpolate",
    "knnidw_interpolate",
    "kriging_interpolate",
]


def _as_xy(name: str, arr: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    return np.ascontiguousarray(out)


def _as_z(name: str, arr: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if out.shape[0] != n:
        raise ValueError(f"{name} length {out.shape[0]} != source length {n}")
    return np.ascontiguousarray(out)


def tin_interpolate(
    src_xy: NDArray[np.float64],
    src_z: NDArray[np.float64],
    query_xy: NDArray[np.float64],
    *,
    extrapolate: Literal["knnidw", "nearest", None] = "knnidw",
    extrapolate_k: int = 3,
    extrapolate_p: float = 1.0,
    extrapolate_rmax: float = 50.0,
) -> NDArray[np.float64]:
    """Delaunay linear interpolation plus optional lidR-style extrapolation."""
    sx = _as_xy("src_xy", src_xy)
    qx = _as_xy("query_xy", query_xy)
    sz = _as_z("src_z", src_z, sx.shape[0])
    if extrapolate not in ("knnidw", "nearest", None):
        raise ValueError("extrapolate must be 'knnidw', 'nearest', or None")
    if qx.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)

    out = np.full((qx.shape[0],), np.nan, dtype=np.float64)
    if sx.shape[0] >= 3:
        try:
            from scipy.interpolate import LinearNDInterpolator

            interp = LinearNDInterpolator(sx, sz, fill_value=np.nan)
            out = np.asarray(interp(qx), dtype=np.float64)
        except Exception:
            out = np.full((qx.shape[0],), np.nan, dtype=np.float64)

    missing = np.isnan(out)
    if missing.any() and extrapolate is not None:
        if extrapolate == "nearest":
            fill = knnidw_interpolate(
                sx,
                sz,
                qx[missing],
                k=1,
                p=1.0,
                rmax=np.inf,
            )
        else:
            fill = knnidw_interpolate(
                sx,
                sz,
                qx[missing],
                k=extrapolate_k,
                p=extrapolate_p,
                rmax=extrapolate_rmax,
            )
        out[missing] = fill
    return out


def knnidw_interpolate(
    src_xy: NDArray[np.float64],
    src_z: NDArray[np.float64],
    query_xy: NDArray[np.float64],
    *,
    k: int = 10,
    p: float = 2.0,
    rmax: float = np.inf,
) -> NDArray[np.float64]:
    """K-nearest inverse-distance weighted interpolation.

    PORT NOTE: the older raster-local helper approximated exact hits via
    ``1 / max(d, 1e-12)^p``. This generic helper treats ``d == 0`` explicitly
    and returns the mean value of coincident source points. Raster cell centers
    rarely coincide with source centers, so existing raster fill semantics are
    practically unchanged; normalization gets a clearer scattered-interpolation
    contract.
    """
    sx = _as_xy("src_xy", src_xy)
    qx = _as_xy("query_xy", query_xy)
    sz = _as_z("src_z", src_z, sx.shape[0])
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (p > 0.0):
        raise ValueError("p must be > 0")
    if not (rmax > 0.0 or np.isinf(rmax)):
        raise ValueError("rmax must be > 0 or inf")
    if qx.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if sx.shape[0] == 0:
        return np.full((qx.shape[0],), np.nan, dtype=np.float64)

    tree = cKDTree(sx)
    k_eff = min(int(k), sx.shape[0])
    d, idx = tree.query(qx, k=k_eff)
    if k_eff == 1:
        d = d[:, None]
        idx = idx[:, None]

    out = np.full((qx.shape[0],), np.nan, dtype=np.float64)
    finite_rmax = float(rmax)
    for i in range(qx.shape[0]):
        row_d = np.asarray(d[i], dtype=np.float64)
        row_idx = np.asarray(idx[i], dtype=np.int64)
        valid = np.isfinite(row_d)
        if np.isfinite(finite_rmax):
            valid &= row_d <= finite_rmax
        if not valid.any():
            continue
        row_d = row_d[valid]
        row_idx = row_idx[valid]
        exact = row_d == 0.0
        if exact.any():
            out[i] = float(np.mean(sz[row_idx[exact]]))
            continue
        weights = 1.0 / (row_d ** p)
        weights /= weights.sum()
        out[i] = float(np.sum(sz[row_idx] * weights))
    return out


def kriging_interpolate(
    src_xy: NDArray[np.float64],
    src_z: NDArray[np.float64],
    query_xy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Universal kriging with the existing raster/gstat-inspired defaults."""
    sx = _as_xy("src_xy", src_xy)
    qx = _as_xy("query_xy", query_xy)
    sz = _as_z("src_z", src_z, sx.shape[0])
    if qx.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if sx.shape[0] < 3:
        return np.full((qx.shape[0],), np.nan, dtype=np.float64)

    from pykrige.uk import UniversalKriging

    # gstat::vgm(0.59, "Sph", 874) -> psill=0.59, range=874, nugget=0.
    uk = UniversalKriging(
        sx[:, 0],
        sx[:, 1],
        sz,
        variogram_model="spherical",
        variogram_parameters=[0.59, 874.0, 0.0],
        drift_terms=["regional_linear"],
        verbose=False,
        enable_plotting=False,
    )
    z_pred, _ = uk.execute("points", qx[:, 0], qx[:, 1])
    return np.asarray(z_pred, dtype=np.float64)
