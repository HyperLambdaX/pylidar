"""Treetop detection and the ``Treetops`` data class (lidR ``locate_trees`` port).

PORT NOTE
---------
Adapted from lidR ``R/locate_trees.R`` (L29-102 dispatch + L73-95 uniqueness),
``R/algorithm-itd.R`` (L72-107 ``lmf`` wrapper), and the Rcpp-side
``src/RcppFunction.cpp`` (L413-440 ``bitmerge`` C kernel). Translation
choices:

* :class:`Treetops` mirrors lidR's ``sf POINT Z`` table: ``x``, ``y``, ``z``
  columns plus a ``tree_id`` column. lidR's ``treeID`` is R ``integer`` for
  ``uniqueness="incremental"`` and R ``double`` for ``"gpstime"`` /
  ``"bitmerge"``. The dataclass field stores either an ``int32`` or a
  ``float64`` numpy array; downstream code dispatches on dtype (user
  decision 2026-05-12).
* :func:`locate_trees_chm` and :func:`locate_trees_points` are the two
  entry points. lidR's ``locate_trees(las, lmf(...))`` wraps both into one
  R-side dispatch via ``raster_as_las``; pylidar already exposes
  ``_core.lmf_chm`` and ``_core.lmf_points`` independently, so this port
  keeps two flat functions instead of recreating the dispatch chain.
* Three uniqueness modes (``incremental`` / ``gpstime`` / ``bitmerge``) are
  implemented in :func:`_assign_tree_ids`. The ``bitmerge`` kernel is a
  literal port of the Rcpp loop at ``src/RcppFunction.cpp:431-435``: scaled
  X/Y → ``int32`` → bitcast to ``uint32`` → pack into ``uint64`` (X high,
  Y low) → bitcast to ``float64``. Z is **not** encoded. Negative scaled
  coordinates ride on two's-complement memcpy, so a Python implementation
  on top of ``numpy.uint32`` / ``numpy.uint64`` round-trips through
  ``view`` is faithful.
* ``bitmerge`` requires the LAS header offset/scale per dimension because
  lidR pulls them straight off the LAS object (``locate_trees.R:83-90``).
  The Treetops dataclass deliberately does **not** carry these; callers
  must pass ``las_offset`` and ``las_scale`` keyword arguments. ``gpstime``
  mode similarly requires the per-treetop GPS timestamps as a separate
  array.
* CHM-level ``gpstime``/``bitmerge`` raise: a CHM has no GPS timestamps,
  and ``bitmerge`` needs LAS scale/offset metadata which a CHM lacks
  unless the user passes them explicitly. ``locate_trees_chm`` accepts
  ``las_offset``/``las_scale`` so ``bitmerge`` works on a CHM derived from
  a known LAS; ``gpstime`` raises ``NotImplementedError`` for CHM input
  with a helpful message pointing at ``locate_trees_points``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import _core
from .raster import RasterLayout

try:  # pyproj is a runtime dep; keep import lazy-friendly for tooling.
    import pyproj
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pylidar.locate_trees requires pyproj — install pylidar default extras"
    ) from exc

__all__ = [
    "Treetops",
    "locate_trees_chm",
    "locate_trees_points",
    "bitmerge",
]


Uniqueness = Literal["incremental", "gpstime", "bitmerge"]


# ────────────────────────────────────────────────────────── Treetops dataclass

@dataclass(frozen=True)
class Treetops:
    """Detected treetops: per-tree (x, y, z, tree_id) plus optional CRS.

    Mirrors lidR's ``sf POINT Z`` output of ``locate_trees``.

    Parameters
    ----------
    x, y, z : (K,) float64 ndarray
        Treetop world coordinates.
    tree_id : (K,) ndarray
        Per-tree identifier. dtype is **mode-dependent**:

        * ``int32`` for ``uniqueness="incremental"`` (1..K, lidR `integer`).
        * ``float64`` for ``uniqueness="gpstime"`` (raw GPS time of apex)
          and ``uniqueness="bitmerge"`` (IEEE-754 reinterpret of the
          uint64 X/Y packing).

        Downstream code (``segment_trees``, ``write_las_with_treeid``,
        ``dalponte2016_from_treetops``) inspects ``tree_id.dtype`` to
        choose its raster output dtype.
    crs : pyproj.CRS | None
        Coordinate reference system, inherited from the source LAS / CHM.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    tree_id: NDArray  # int32 or float64 — see docstring
    crs: Optional["pyproj.CRS"] = None

    def __post_init__(self) -> None:
        # frozen=True forbids attribute assignment, so use object.__setattr__
        # for normalization. Defensive: callers may hand us views.
        for name in ("x", "y", "z"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.dtype != np.float64 or arr.ndim != 1:
                raise TypeError(
                    f"Treetops.{name} must be a 1-D float64 ndarray, got "
                    f"{type(arr).__name__} dtype={getattr(arr, 'dtype', None)} "
                    f"shape={getattr(arr, 'shape', None)}"
                )
        tid = self.tree_id
        if not isinstance(tid, np.ndarray) or tid.ndim != 1:
            raise TypeError("Treetops.tree_id must be a 1-D ndarray")
        if tid.dtype not in (np.int32, np.float64):
            raise TypeError(
                "Treetops.tree_id must be int32 (incremental) or float64 "
                f"(gpstime/bitmerge), got {tid.dtype}"
            )
        if not (self.x.shape == self.y.shape == self.z.shape == self.tree_id.shape):
            raise ValueError(
                f"Treetops fields must share length, got x={self.x.shape}, "
                f"y={self.y.shape}, z={self.z.shape}, tree_id={self.tree_id.shape}"
            )
        if self.crs is not None and not isinstance(self.crs, pyproj.CRS):
            raise TypeError("Treetops.crs must be a pyproj.CRS or None")

    @property
    def n(self) -> int:
        """Number of detected treetops."""
        return int(self.x.shape[0])

    def __len__(self) -> int:
        return self.n


# ─────────────────────────────────────────── bitmerge (1:1 lidR Rcpp port)

def bitmerge(
    x_int32: NDArray[np.int32], y_int32: NDArray[np.int32]
) -> NDArray[np.float64]:
    """Pack two int32 arrays into double via uint64 numeric cast.

    1:1 port of lidR ``src/RcppFunction.cpp::bitmerge`` (L413-440). Each
    pair ``(xi, yi)`` is **bitcast** to ``uint32`` (memcpy preserves bit
    pattern, so negative values ride two's-complement), packed into a
    ``uint64`` as ``(uint32(xi) << 32) | uint32(yi)``, then **numerically
    cast** to ``float64`` via ``static_cast<double>(uint64_t)`` semantics.

    Note that the cast is value-preserving up to ``2**53`` and rounds
    above; for the upper end of the uint64 range, multiple distinct
    packings can collide on the same double. lidR accepts this; the doc
    string at lidR ``man-roxygen/section-uniqueness.R:36`` describes a
    bit-reinterpret instead, but the actual C++ code uses
    ``static_cast<double>``. Empirically: ``bitmerge([1], [2]) ==
    [4294967298.0]``.
    """
    if x_int32.shape != y_int32.shape:
        raise ValueError("bitmerge: x and y must share shape")
    if x_int32.dtype != np.int32 or y_int32.dtype != np.int32:
        raise TypeError("bitmerge: both inputs must be int32")
    # Bitcast int32 → uint32 (preserves bit pattern; numpy `view` does the
    # memcpy-equivalent on a contiguous buffer).
    xu = np.ascontiguousarray(x_int32).view(np.uint32)
    yu = np.ascontiguousarray(y_int32).view(np.uint32)
    packed = (xu.astype(np.uint64) << np.uint64(32)) | yu.astype(np.uint64)
    # Numeric cast uint64 → float64, matching `static_cast<double>` at
    # lidR `RcppFunction.cpp:435`. NOT a bit reinterpret.
    return packed.astype(np.float64)


# ─────────────────────────────────────────── uniqueness ID assignment

def _assign_tree_ids(
    *,
    uniqueness: Uniqueness,
    n: int,
    gpstime: Optional[NDArray[np.float64]] = None,
    x_world: Optional[NDArray[np.float64]] = None,
    y_world: Optional[NDArray[np.float64]] = None,
    las_offset: Optional[Tuple[float, float]] = None,
    las_scale: Optional[Tuple[float, float]] = None,
) -> NDArray:
    """Compute per-treetop ID array per the uniqueness contract.

    Returns int32 array (incremental) or float64 array (gpstime / bitmerge).
    """
    if uniqueness == "incremental":
        return np.arange(1, n + 1, dtype=np.int32)

    if uniqueness == "gpstime":
        if gpstime is None:
            raise ValueError(
                "uniqueness='gpstime' requires the per-treetop gpstime array"
            )
        if gpstime.shape != (n,):
            raise ValueError(
                f"gpstime must have shape ({n},), got {gpstime.shape}"
            )
        if gpstime.dtype != np.float64:
            raise TypeError("gpstime must be float64")
        return gpstime.astype(np.float64, copy=True)

    if uniqueness == "bitmerge":
        if x_world is None or y_world is None:
            raise ValueError(
                "uniqueness='bitmerge' requires per-treetop x/y world coords "
                "(internal call bug if you see this from a user-facing entry)"
            )
        if las_offset is None or las_scale is None:
            raise ValueError(
                "uniqueness='bitmerge' requires las_offset=(xoff, yoff) and "
                "las_scale=(xscale, yscale) keyword arguments — these come "
                "from the LAS header (lidR locate_trees.R:83-90)"
            )
        xo, yo = float(las_offset[0]), float(las_offset[1])
        xs, ys = float(las_scale[0]), float(las_scale[1])
        if not (xs > 0.0 and ys > 0.0):
            raise ValueError("las_scale entries must be > 0")
        # lidR R wrapper uses `as.integer((x - xoffset) / xscale)`. R's
        # `as.integer` truncates toward zero (matches C int cast). Replicate
        # with `np.trunc` then cast.
        xs_arr = np.trunc((x_world - xo) / xs).astype(np.int32)
        ys_arr = np.trunc((y_world - yo) / ys).astype(np.int32)
        return bitmerge(xs_arr, ys_arr)

    raise ValueError(
        f"unknown uniqueness {uniqueness!r}; expected one of "
        "'incremental', 'gpstime', 'bitmerge'"
    )


# ────────────────────────────────────────────────────────── locate_trees_chm

def locate_trees_chm(
    chm: NDArray[np.float64],
    layout: RasterLayout,
    *,
    ws: float,
    hmin: float = 2.0,
    shape: Literal["circular", "square"] = "circular",
    uniqueness: Uniqueness = "incremental",
    las_offset: Optional[Tuple[float, float]] = None,
    las_scale: Optional[Tuple[float, float]] = None,
) -> Treetops:
    """Detect treetops on a CHM via lidR-style local-maximum filter.

    Wraps :func:`pylidar.segmentation.lmf_chm` (which calls ``_core.lmf_chm``):
    the kernel returns ``(K, 2)`` row/col indices; this function reverses
    them through ``layout`` for world X/Y, samples ``chm[row, col]`` for Z,
    and assigns IDs per ``uniqueness``.

    Parameters
    ----------
    chm : (H, W) float64 ndarray
        Canopy height model; NaN cells excluded from local-max search.
    layout : RasterLayout
        Spatial extent + CRS for the CHM. Used to invert (row, col) → (x, y)
        AND to convert ``ws`` from world units to pixel units.
    ws : float
        Window size in **world coordinate units** (matches lidR
        ``lmf(ws=...)`` semantics — lidR converts a CHM to a pseudo-LAS
        first, so its ``ws`` is always in world units). The wrapper divides
        by ``layout.xres`` to translate to the pixel-unit ``ws`` that
        ``_core.lmf_chm`` expects. ``layout.xres`` must equal
        ``layout.yres`` (square pixels); rectangular pixels would require
        an elliptical kernel that ``_core.lmf_chm`` does not implement.
    hmin : float, default 2.0
        Minimum height for a cell to count as a treetop.
    shape : {'circular', 'square'}, default 'circular'
    uniqueness : {'incremental', 'gpstime', 'bitmerge'}
        ID assignment strategy. ``'gpstime'`` raises ``NotImplementedError``
        because a CHM does not carry GPS timestamps. ``'bitmerge'`` requires
        ``las_offset`` and ``las_scale`` from the source LAS header.
    las_offset, las_scale : (float, float), optional
        LAS header X/Y offsets and scale factors. Required only for
        ``uniqueness='bitmerge'``.

    Returns
    -------
    Treetops
    """
    if shape not in ("circular", "square"):
        raise ValueError("shape must be 'circular' or 'square'")
    if uniqueness == "gpstime":
        raise NotImplementedError(
            "uniqueness='gpstime' is not supported for CHM-based detection — "
            "a CHM has no GPS timestamps. Use locate_trees_points on the "
            "underlying point cloud instead."
        )
    if not isinstance(chm, np.ndarray) or chm.dtype != np.float64 or chm.ndim != 2:
        raise TypeError(
            "locate_trees_chm: chm must be a 2-D float64 ndarray"
        )
    if chm.shape != layout.shape:
        raise ValueError(
            f"locate_trees_chm: chm.shape {chm.shape} must match "
            f"layout.shape {layout.shape}"
        )
    if not (ws > 0.0):
        raise ValueError("locate_trees_chm: ws must be > 0")
    if layout.xres != layout.yres:
        raise NotImplementedError(
            "locate_trees_chm requires square pixels (layout.xres == "
            f"layout.yres); got xres={layout.xres}, yres={layout.yres}. "
            "Resample the CHM or open a follow-up if rectangular pixels "
            "are needed."
        )

    # lidR's `lmf(ws=N)` is in world units even for raster CHMs (because
    # internally it goes through raster_as_las). _core.lmf_chm takes ws in
    # pixel units. Convert here so the user-facing API matches lidR.
    ws_pixels = float(ws) / float(layout.xres)
    # _core.lmf_chm returns (K, 2) int32 row/col pairs.
    rc = _core.lmf_chm(chm=chm, ws=ws_pixels, hmin=hmin, shape=shape)
    rows = rc[:, 0].astype(np.int64)
    cols = rc[:, 1].astype(np.int64)

    x, y = layout.rowcol_to_cell_xy(rows, cols)
    # rowcol_to_cell_xy returns float64 already; ensure ascontiguousarray for
    # downstream views (bitmerge view requires it).
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(chm[rows, cols], dtype=np.float64)

    tree_id = _assign_tree_ids(
        uniqueness=uniqueness,
        n=int(rows.shape[0]),
        x_world=x,
        y_world=y,
        las_offset=las_offset,
        las_scale=las_scale,
    )
    return Treetops(x=x, y=y, z=z, tree_id=tree_id, crs=layout.crs)


# ─────────────────────────────────────────────────────── locate_trees_points

def locate_trees_points(
    xyz: NDArray[np.float64],
    *,
    ws: Union[float, NDArray[np.float64], Callable[[float], float]],
    hmin: float = 2.0,
    shape: Literal["circular", "square"] = "circular",
    uniqueness: Uniqueness = "incremental",
    gpstime: Optional[NDArray[np.float64]] = None,
    las_offset: Optional[Tuple[float, float]] = None,
    las_scale: Optional[Tuple[float, float]] = None,
    crs: Optional["pyproj.CRS"] = None,
) -> Treetops:
    """Detect treetops on a point cloud via lidR-style local-maximum filter.

    Wraps :func:`pylidar.segmentation.lmf_points` (which calls
    ``_core.lmf_points``): the kernel returns an ``(N,)`` bool mask; this
    function selects ``xyz[mask]`` and assigns IDs per ``uniqueness``.

    Parameters
    ----------
    xyz : (N, 3) float64 ndarray
        Point cloud (x, y, z).
    ws : float | (N,) float64 ndarray | callable
        Window size — see :func:`pylidar.segmentation.lmf_points`.
    hmin : float, default 2.0
    shape : {'circular', 'square'}, default 'circular'
    uniqueness : {'incremental', 'gpstime', 'bitmerge'}
    gpstime : (N,) float64 ndarray, optional
        Per-input-point GPS timestamps. Required for ``uniqueness='gpstime'``;
        the per-treetop subset is ``gpstime[mask]``.
    las_offset, las_scale : (float, float), optional
        Required for ``uniqueness='bitmerge'`` (LAS header values).
    crs : pyproj.CRS, optional
        Inherited from the source LAS; carried through onto the Treetops.

    Returns
    -------
    Treetops
    """
    # Defer parameter validation to lmf_points (already strict-checked).
    from .segmentation import lmf_points

    mask = lmf_points(xyz=xyz, ws=ws, hmin=hmin, shape=shape)
    if mask.shape != (xyz.shape[0],):
        # Defensive: lmf_points contract is (N,) bool, but pin it.
        raise RuntimeError(
            f"lmf_points returned shape {mask.shape}, expected ({xyz.shape[0]},)"
        )

    sel = xyz[mask]
    x = np.ascontiguousarray(sel[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(sel[:, 1], dtype=np.float64)
    z = np.ascontiguousarray(sel[:, 2], dtype=np.float64)

    gps_subset: Optional[NDArray[np.float64]] = None
    if uniqueness == "gpstime":
        if gpstime is None:
            raise ValueError(
                "uniqueness='gpstime' requires gpstime=<(N,) float64> matching "
                "xyz; lidR locate_trees.R:78 reads `las@data[['gpstime']][res]`"
            )
        if gpstime.shape != (xyz.shape[0],):
            raise ValueError(
                f"gpstime must have shape ({xyz.shape[0]},), got {gpstime.shape}"
            )
        if gpstime.dtype != np.float64:
            raise TypeError("gpstime must be float64")
        # lidR locate_trees.R:44-45 rejects an all-zero gpstime as
        # "not populated". Match: check the full input array, not just
        # the apex subset (matches lidR's `fast_countequal(gpstime, 0)
        # == npoints` check on the LAS object).
        if xyz.shape[0] > 0 and not np.any(gpstime != 0.0):
            raise ValueError(
                "Impossible to compute unique IDs using gpstime: gpstime is "
                "not populated (all-zero). lidR locate_trees.R:44-45."
            )
        gps_subset = np.ascontiguousarray(gpstime[mask], dtype=np.float64)

    tree_id = _assign_tree_ids(
        uniqueness=uniqueness,
        n=int(x.shape[0]),
        gpstime=gps_subset,
        x_world=x,
        y_world=y,
        las_offset=las_offset,
        las_scale=las_scale,
    )
    return Treetops(x=x, y=y, z=z, tree_id=tree_id, crs=crs)
