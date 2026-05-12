"""Canopy-height-model rasterization layer (lidR ``rasterize_canopy`` port).

PORT NOTE
---------
Adapted from lidR ``R/algorithm-dsm.R`` (p2r L42-93, pitfree L209-306,
dsmtin L140-144, spikefree L342-403), ``R/rasterize_canopy.R`` (L9-35),
``R/utils_raster.R`` (L55-77, coord convention), and ``R/algorithm-dtm.R``
(L129-161, kriging defaults). Translation choices:

* :class:`RasterLayout` mirrors lidR's ``layout`` list (xmin, ymax, xres,
  yres, ncol, nrow, crs) — image-style upper-left origin at ``(xmin, ymax)``,
  row index increases downward (cartographic Y inverted). The cell mapping
  is the verbatim ``floor((x-xmin)/xres)`` / ``floor((ymax-y)/yres)`` pair
  with right/bottom edge clamping; cell centers are
  ``(xmin + (col + 0.5) * xres, ymax - (row + 0.5) * yres)``. The CRS slot
  carries a :class:`pyproj.CRS` (``None`` if absent), per the user's
  2026-05-11 Q2 decision.
* :func:`rasterize_canopy_p2r` aggregates by **max** (lidR
  ``algorithm-dsm.R:57`` ``"max"``); empty cells are ``np.nan``. Subcircle
  **replaces** each input point with 8 equiangular satellites at radii
  ``cos/sin(i * 2π/8)`` for ``i ∈ [0, 8)``, clipped to the layout bbox to
  avoid extrapolation (``algorithm-dsm.R::subcircle`` L420-433 — the helper
  emits only the satellites, the original point is dropped). The
  mm-precision ``round(z, 3)`` quantization is applied at the function
  boundary (``rasterize_canopy.R:28``). First returns are **not** enforced
  for p2r.
* :func:`rasterize_canopy_pitfree` enforces ``ReturnNumber == 1`` with the
  verbatim lidR error strings (``algorithm-dsm.R:233-234``). The TIN is
  built with :class:`scipy.spatial.Delaunay`; triangles whose longest edge
  exceeds the active ``max_edge`` are dropped before interpolation. Layered
  thresholds are merged via :func:`np.fmax`. Numerical divergence from
  RTriangle's tie-break in degenerate equidistant cases is expected and
  documented; fixtures avoid those topologies (port-notes §3 fixture
  topology trap).
* :func:`rasterize_canopy_dsmtin` is the lidR one-liner alias
  (``algorithm-dsm.R:140-144``: ``dsmtin(max_edge, highest) = pitfree(0,
  c(max_edge, 0), 0, highest)``).
* :func:`rasterize_canopy_spikefree` is a stub that raises
  :class:`NotImplementedError`. lidR's implementation lives in C++
  (``C_spikefree``, Fischer et al. 2024 iterative TIN refinement) and was
  judged out-of-scope for this phase; future work can port the C kernel.
* ``na_fill`` for p2r supports ``None`` (NaN preserved), ``"tin"`` (Delaunay
  + linear interpolation on non-NaN cell centers), ``"knnidw"`` (k-nearest
  inverse-distance-weighted), and ``"kriging"`` (pykrige
  :class:`UniversalKriging` with a Spherical variogram parameterized as
  lidR's ``gstat::vgm(0.59, "Sph", 874)`` and a regional-linear drift, per
  the user's 2026-05-11 Q4 decision). All three methods restrict
  interpolation to NaN cells whose centers lie inside the convex hull of
  the input ``xyz`` (1:1 with lidR's ``st_intersection(where, hull)`` at
  ``algorithm-dsm.R:71``); cells outside the hull stay NaN. **v0
  simplifications**: (1) the hull is **not** buffered by ``res`` like
  lidR's ``st_buffer(hull, dist=res)``, since shapely is not a runtime
  dep — cells on the hull boundary may be omitted by a fraction of a
  cell. (2) Kriging uses the **full** UK system; lidR's
  ``gstat::krige(..., nmax=10)`` k-NN limit is dropped because pykrige's
  ``UniversalKriging.execute`` has no equivalent (only OrdinaryKriging
  exposes ``n_closest_points``). For RAM-bounded production grids, future
  work can pre-filter source points per-target. The kriging output is
  **not** numerically identical to gstat — different solvers, different
  conditioning — but the model and trend are the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, cKDTree

try:  # pyproj is a runtime dep but keep import lazy-friendly for tooling.
    import pyproj
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pylidar.raster requires pyproj — install pylidar with the default extras"
    ) from exc

__all__ = [
    "RasterLayout",
    "rasterize_canopy_p2r",
    "rasterize_canopy_pitfree",
    "rasterize_canopy_dsmtin",
    "rasterize_canopy_spikefree",
]


# ---------------------------------------------------------------- RasterLayout

@dataclass(frozen=True)
class RasterLayout:
    """Raster grid extent + resolution + CRS.

    Mirrors lidR's ``layout`` list: upper-left origin at ``(xmin, ymax)``;
    row index increases downward (image / cartographic-inverted Y). Cell
    centers live at ``(xmin + (col + 0.5) * xres, ymax - (row + 0.5) * yres)``.
    """

    xmin: float
    ymax: float
    xres: float
    yres: float
    ncol: int
    nrow: int
    crs: Optional["pyproj.CRS"] = None

    @property
    def xmax(self) -> float:
        return self.xmin + self.ncol * self.xres

    @property
    def ymin(self) -> float:
        return self.ymax - self.nrow * self.yres

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrow, self.ncol)

    @classmethod
    def from_extent(
        cls,
        xyz: NDArray[np.float64],
        res: float,
        *,
        crs: Optional["pyproj.CRS"] = None,
    ) -> "RasterLayout":
        """Build a layout that wraps the bounding box of ``xyz`` at ``res``.

        Aligns extent so ``ncol = ceil((xmax - xmin) / res)`` and
        ``nrow = ceil((ymax - ymin) / res)`` covers every point. Lazy-aligned
        bbox (no ``st_adjust_bbox`` snap-to-grid; that's the LAScatalog
        Phase-6 concern). Inputs ``xyz`` may be empty → raises ValueError.
        """
        if xyz.shape[0] == 0:
            raise ValueError("from_extent: xyz must have at least one point")
        xmin = float(xyz[:, 0].min())
        xmax = float(xyz[:, 0].max())
        ymin = float(xyz[:, 1].min())
        ymax = float(xyz[:, 1].max())
        ncol = max(int(np.ceil((xmax - xmin) / res)), 1)
        nrow = max(int(np.ceil((ymax - ymin) / res)), 1)
        return cls(xmin=xmin, ymax=ymax, xres=res, yres=res, ncol=ncol, nrow=nrow, crs=crs)

    def cell_xy_to_rowcol(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Map world ``(x, y)`` → image ``(row, col)``.

        Right edge: ``x == xmax`` clamps to ``col = ncol - 1``.
        Bottom edge: ``y == ymin`` clamps to ``row = nrow - 1``.
        Out-of-extent points yield row/col outside ``[0, nrow/ncol)``; the
        caller decides how to gate.
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        col = np.floor((x - self.xmin) / self.xres).astype(np.int64)
        row = np.floor((self.ymax - y) / self.yres).astype(np.int64)
        # Edge clamping per lidR utils_raster.R:71-74
        col[x == self.xmax] = self.ncol - 1
        row[y == self.ymin] = self.nrow - 1
        return row, col

    def rowcol_to_cell_xy(
        self, row: NDArray[np.int64], col: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map ``(row, col)`` → cell-center world coords."""
        row = np.asarray(row, dtype=np.int64)
        col = np.asarray(col, dtype=np.int64)
        x = self.xmin + (col + 0.5) * self.xres
        y = self.ymax - (row + 0.5) * self.yres
        return x, y


# ---------------------------------------------------------------- p2r

def rasterize_canopy_p2r(
    xyz: NDArray[np.float64],
    layout: RasterLayout,
    *,
    subcircle: float = 0.0,
    na_fill: Optional[Literal["tin", "knnidw", "kriging"]] = None,
) -> NDArray[np.float64]:
    """Points-to-raster CHM (lidR ``p2r``).

    Aggregates by max within each cell; empty cells → NaN. Optional
    ``subcircle > 0`` expands each point into 8 equiangular satellites at
    that radius (clipped to layout bbox). The mm-precision ``round(z, 3)``
    is applied at function exit. First returns are NOT enforced.

    ``na_fill`` populates NaN cells via TIN linear interp, k-NN IDW, or
    pykrige UniversalKriging (Spherical variogram, regional-linear drift).
    """
    _check_xyz(xyz)
    if subcircle < 0:
        raise ValueError("subcircle must be >= 0")

    pts_x, pts_y, pts_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if subcircle > 0 and xyz.shape[0] > 0:
        # 1:1 with lidR ``algorithm-dsm.R::subcircle`` (L420-433): the helper
        # **replaces** each input point with 8 equiangular satellites; the
        # original is dropped. Bbox clip mirrors data.table::between filter.
        pts_x, pts_y, pts_z = _subcircle_satellites(pts_x, pts_y, pts_z, subcircle, layout)

    chm = _rasterize_max(pts_x, pts_y, pts_z, layout)

    if na_fill is not None:
        chm = _apply_na_fill(chm, layout, xyz, na_fill)

    # mm quantization at output (rasterize_canopy.R:28); preserves NaN.
    finite = np.isfinite(chm)
    chm[finite] = np.round(chm[finite], 3)
    return chm


def _subcircle_satellites(
    pts_x: NDArray[np.float64],
    pts_y: NDArray[np.float64],
    pts_z: NDArray[np.float64],
    radius: float,
    layout: RasterLayout,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Replace each input point with 8 equiangular satellites at ``radius``,
    clipped to the layout bbox. Originals are NOT preserved (1:1 with
    ``algorithm-dsm.R::subcircle`` L420-433)."""
    angles = np.arange(8) * (2.0 * np.pi / 8.0)
    cs, ss = np.cos(angles), np.sin(angles)
    sat_x = (pts_x[:, None] + radius * cs[None, :]).ravel()
    sat_y = (pts_y[:, None] + radius * ss[None, :]).ravel()
    sat_z = np.repeat(pts_z, 8)
    in_bbox = (
        (sat_x >= layout.xmin)
        & (sat_x <= layout.xmax)
        & (sat_y >= layout.ymin)
        & (sat_y <= layout.ymax)
    )
    return sat_x[in_bbox], sat_y[in_bbox], sat_z[in_bbox]


def _rasterize_max(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    layout: RasterLayout,
) -> NDArray[np.float64]:
    chm = np.full(layout.shape, np.nan, dtype=np.float64)
    if x.size == 0:
        return chm
    row, col = layout.cell_xy_to_rowcol(x, y)
    in_grid = (row >= 0) & (row < layout.nrow) & (col >= 0) & (col < layout.ncol)
    if not in_grid.any():
        return chm
    row, col, z = row[in_grid], col[in_grid], z[in_grid]
    flat_idx = row * layout.ncol + col
    flat = chm.ravel()
    # np.maximum.at handles NaN start by overwriting via fmax-like path.
    # Use -inf init then convert back to NaN for cells that stayed -inf.
    flat[:] = -np.inf
    np.maximum.at(flat, flat_idx, z)
    chm = flat.reshape(layout.shape)
    chm = np.where(np.isfinite(chm), chm, np.nan)
    return chm


# ---------------------------------------------------------------- na_fill

def _apply_na_fill(
    chm: NDArray[np.float64],
    layout: RasterLayout,
    xyz: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    if method not in ("tin", "knnidw", "kriging"):
        raise ValueError(f"unknown na_fill method: {method!r}")

    nan_mask = np.isnan(chm)
    if not nan_mask.any():
        return chm
    finite_mask = ~nan_mask
    if not finite_mask.any():
        return chm

    rows, cols = np.indices(chm.shape)
    src_x, src_y = layout.rowcol_to_cell_xy(rows[finite_mask], cols[finite_mask])
    src_z = chm[finite_mask]
    nan_rows = rows[nan_mask]
    nan_cols = cols[nan_mask]
    tgt_x, tgt_y = layout.rowcol_to_cell_xy(nan_rows, nan_cols)

    # Restrict interpolation to NaN cell centers inside the convex hull of
    # the input xyz (1:1 with lidR algorithm-dsm.R:71
    # ``st_intersection(where, hull)``). v0 simplification: no res-buffered
    # expansion (shapely not a dep) — see PORT NOTE.
    in_hull = _convex_hull_mask(xyz[:, 0:2], tgt_x, tgt_y)
    if not in_hull.any():
        return chm

    fill_x = tgt_x[in_hull]
    fill_y = tgt_y[in_hull]

    if method == "tin":
        filled = _tin_interp(src_x, src_y, src_z, fill_x, fill_y)
    elif method == "knnidw":
        filled = _knn_idw(src_x, src_y, src_z, fill_x, fill_y, k=10, p=2.0)
    else:  # kriging
        filled = _kriging(src_x, src_y, src_z, fill_x, fill_y)

    out = chm.copy()
    # Index into the (rows,cols) of NaN cells that survived the hull mask.
    out[nan_rows[in_hull], nan_cols[in_hull]] = filled
    return out


def _convex_hull_mask(
    points_xy: NDArray[np.float64],
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Boolean mask over targets — True where target is inside the convex
    hull of ``points_xy``. Degenerate hulls (<3 points or collinear) fall
    back to "all True" (lidR's hull would be degenerate too; we don't try
    to mimic gstat's behaviour on degenerate input)."""
    if points_xy.shape[0] < 3:
        return np.ones(tx.shape, dtype=np.bool_)
    try:
        tri = Delaunay(points_xy)
    except Exception:
        # Collinear / coplanar — fall back to no restriction.
        return np.ones(tx.shape, dtype=np.bool_)
    return tri.find_simplex(np.column_stack([tx, ty])) >= 0


def _tin_interp(
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    sz: NDArray[np.float64],
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
) -> NDArray[np.float64]:
    if sx.size < 3:
        return np.full(tx.shape, np.nan, dtype=np.float64)
    # Use scipy LinearNDInterpolator under the hood.
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(np.column_stack([sx, sy]), sz, fill_value=np.nan)
    return interp(np.column_stack([tx, ty]))


def _knn_idw(
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    sz: NDArray[np.float64],
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    *,
    k: int,
    p: float,
) -> NDArray[np.float64]:
    if sx.size == 0:
        return np.full(tx.shape, np.nan, dtype=np.float64)
    tree = cKDTree(np.column_stack([sx, sy]))
    k_eff = min(k, sx.size)
    d, idx = tree.query(np.column_stack([tx, ty]), k=k_eff)
    if k_eff == 1:
        d = d[:, None]
        idx = idx[:, None]
    # Guard against d == 0 (target coincides with source).
    weights = 1.0 / np.maximum(d, 1e-12) ** p
    weights /= weights.sum(axis=1, keepdims=True)
    return (sz[idx] * weights).sum(axis=1)


def _kriging(
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    sz: NDArray[np.float64],
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
) -> NDArray[np.float64]:
    if sx.size < 3:
        return np.full(tx.shape, np.nan, dtype=np.float64)
    from pykrige.uk import UniversalKriging

    # gstat::vgm(0.59, "Sph", 874) → psill=0.59, range=874, nugget=0
    # universal kriging with linear (regional) drift = Z ~ X + Y
    uk = UniversalKriging(
        sx,
        sy,
        sz,
        variogram_model="spherical",
        variogram_parameters=[0.59, 874.0, 0.0],
        drift_terms=["regional_linear"],
        verbose=False,
        enable_plotting=False,
    )
    z_pred, _ = uk.execute("points", tx, ty)
    return np.asarray(z_pred, dtype=np.float64)


# ---------------------------------------------------------------- pitfree / dsmtin

_PITFREE_NO_RETURNNUMBER = "No attribute 'ReturnNumber' found"
_PITFREE_NO_FIRST_RETURN = "No first returns found"


def rasterize_canopy_pitfree(
    xyz: NDArray[np.float64],
    return_number: NDArray[np.integer],
    layout: RasterLayout,
    *,
    thresholds: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0),
    max_edge: tuple[float, float] = (0.0, 1.0),
    subcircle: float = 0.0,
    highest: bool = True,
) -> NDArray[np.float64]:
    """Khosravipour pitfree CHM (lidR ``pitfree``).

    Layered TIN: at each ascending threshold, filter ``Z >= t``, build
    Delaunay triangulation, drop triangles whose longest edge exceeds
    ``max_edge[i]`` (or ``max_edge[1]`` for ``i > 0``), linearly interpolate
    onto cell centers, and merge layers via per-cell max. First returns are
    enforced; ``return_number`` is required.
    """
    _check_xyz(xyz)
    _enforce_first_returns(xyz, return_number)

    if len(thresholds) > 1 and len(max_edge) < 2:
        raise ValueError("'max_edge' should contain 2 numbers")

    rn = np.asarray(return_number)
    first = (rn == 1)
    pts = xyz[first]

    if subcircle > 0:
        # 1:1 with algorithm-dsm.R::subcircle (L420-433): replace each point
        # with 8 satellites, drop original.
        sx, sy, sz = _subcircle_satellites(
            pts[:, 0], pts[:, 1], pts[:, 2], subcircle, layout
        )
        pts = np.column_stack([sx, sy, sz])

    if highest:
        from .decimate import decimate_highest
        keep = decimate_highest(pts, layout)
        pts = pts[keep]
        # Match lidR algorithm-dsm.R:265: stop if decimation leaves < 3 points.
        if pts.shape[0] < 3:
            raise ValueError("There are not enought points to triangulate.")

    # Layered TIN merge.
    out = np.full(layout.shape, np.nan, dtype=np.float64)
    grid_rows, grid_cols = np.indices(layout.shape)
    cell_x, cell_y = layout.rowcol_to_cell_xy(grid_rows.ravel(), grid_cols.ravel())
    sorted_thresholds = sorted(thresholds)
    for i, thr in enumerate(sorted_thresholds):
        # 1:1 with algorithm-dsm.R:281 — branch on the threshold VALUE, not
        # the layer index. lidR: ``edge <- if (th == 0) max_edge[1] else max_edge[2]``.
        edge_limit = max_edge[0] if thr == 0.0 else max_edge[1]

        # Match lidR algorithm-dsm.R:284 — only process layers with > 3
        # points above the threshold (note: lidR uses fast_countover which
        # is strict-greater, so >= matches `Z >= th` filter and `> 3` count).
        layer_pts = pts[pts[:, 2] >= thr]
        if layer_pts.shape[0] <= 3:
            continue
        layer = _tin_layer(
            layer_pts[:, 0], layer_pts[:, 1], layer_pts[:, 2],
            cell_x, cell_y,
            edge_limit=edge_limit,
        ).reshape(layout.shape)

        if i == 0 and not np.isfinite(layer).any():
            raise ValueError(
                "Interpolation failed in the first layer (NAs everywhere). "
                "Maybe there are too few points."
            )

        out = np.fmax(out, layer)

    if not np.isfinite(out).any():
        raise ValueError(
            "Interpolation failed (NAs everywhere). Input parameters might be wrong."
        )

    finite = np.isfinite(out)
    out[finite] = np.round(out[finite], 3)
    return out


def rasterize_canopy_dsmtin(
    xyz: NDArray[np.float64],
    return_number: NDArray[np.integer],
    layout: RasterLayout,
    *,
    max_edge: float = 0.0,
    highest: bool = True,
) -> NDArray[np.float64]:
    """Pure TIN DSM. Equivalent to ``pitfree(thresholds=[0], max_edge=[max_edge, 0])``
    per lidR ``algorithm-dsm.R:140-144``."""
    return rasterize_canopy_pitfree(
        xyz,
        return_number,
        layout,
        thresholds=(0.0,),
        max_edge=(max_edge, 0.0),
        subcircle=0.0,
        highest=highest,
    )


def rasterize_canopy_spikefree(*args, **kwargs) -> NDArray[np.float64]:
    """Khosravipour spike-free CHM. **Not implemented in this phase.**

    lidR's implementation lives in C++ (``C_spikefree``, Fischer et al.
    2024 iterative TIN refinement). Porting the C kernel was deferred from
    Phase 3 — open as a follow-up if a user lands a real need.
    """
    raise NotImplementedError(
        "rasterize_canopy_spikefree is not implemented in this phase. "
        "Use rasterize_canopy_pitfree for similar gap-filling behavior."
    )


def _enforce_first_returns(xyz: NDArray[np.float64], rn: Optional[NDArray]) -> None:
    if rn is None:
        raise ValueError(f"{_PITFREE_NO_RETURNNUMBER} (pitfree/dsmtin require ReturnNumber)")
    rn = np.asarray(rn)
    if rn.shape != (xyz.shape[0],):
        raise ValueError(
            f"return_number must have shape ({xyz.shape[0]},), got {rn.shape}"
        )
    if (rn == 1).sum() == 0:
        raise ValueError(f"{_PITFREE_NO_FIRST_RETURN} (pitfree/dsmtin)")


def _tin_layer(
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    sz: NDArray[np.float64],
    tx: NDArray[np.float64],
    ty: NDArray[np.float64],
    *,
    edge_limit: float,
) -> NDArray[np.float64]:
    """Delaunay TIN linear interpolation with optional max-edge trim.

    ``edge_limit <= 0`` disables trimming (lidR convention: max_edge=0 means
    unlimited).
    """
    points = np.column_stack([sx, sy])
    if points.shape[0] < 3:
        return np.full(tx.shape, np.nan, dtype=np.float64)
    try:
        tri = Delaunay(points)
    except Exception:
        return np.full(tx.shape, np.nan, dtype=np.float64)

    # Identify triangles whose longest edge exceeds the limit.
    if edge_limit > 0:
        simplex_pts = points[tri.simplices]  # (T, 3, 2)
        edges = np.stack([
            simplex_pts[:, 1] - simplex_pts[:, 0],
            simplex_pts[:, 2] - simplex_pts[:, 1],
            simplex_pts[:, 0] - simplex_pts[:, 2],
        ], axis=1)  # (T, 3, 2)
        edge_lens = np.linalg.norm(edges, axis=2)  # (T, 3)
        max_edges = edge_lens.max(axis=1)  # (T,)
        keep_simplex = max_edges <= edge_limit
    else:
        keep_simplex = np.ones(tri.simplices.shape[0], dtype=bool)

    # Find the simplex containing each target point.
    targets = np.column_stack([tx, ty])
    simplex_idx = tri.find_simplex(targets)

    out = np.full(tx.shape, np.nan, dtype=np.float64)
    inside = simplex_idx >= 0
    if not inside.any():
        return out
    valid = inside & keep_simplex[simplex_idx]
    if not valid.any():
        return out

    # Barycentric interpolation: scipy provides simplex transforms.
    transform = tri.transform[simplex_idx[valid]]  # (M, 3, 2) — first 2 rows = bary, last = origin
    rel = targets[valid] - transform[:, 2]
    bary = np.einsum("ijk,ik->ij", transform[:, :2, :], rel)  # (M, 2)
    bary = np.column_stack([bary, 1.0 - bary.sum(axis=1)])  # (M, 3)
    z_at_simplices = sz[tri.simplices[simplex_idx[valid]]]  # (M, 3)
    out[valid] = (bary * z_at_simplices).sum(axis=1)
    return out


# ---------------------------------------------------------------- helpers

def _check_xyz(xyz: NDArray[np.float64]) -> None:
    if not isinstance(xyz, np.ndarray):
        raise TypeError("xyz must be a numpy ndarray")
    if xyz.dtype != np.float64:
        raise TypeError("xyz must be float64")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
