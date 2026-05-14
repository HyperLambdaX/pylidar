"""Public ITS algorithms for pylidar.

Per spec §3, the user-facing API lives here. C++ algorithms come from
`pylidar._core`; pure-Python algorithms (silva2016, watershed) live in this
module directly. All function arguments are keyword-only.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from . import _core

__all__ = [
    "dalponte2016",
    "silva2016",
    "li2012",
    "lmf_points",
    "lmf_chm",
    "chm_smooth",
    "watershed",
    # Phase 4 high-level wrappers (lidR-shape ITS API on top of Treetops):
    "dalponte2016_from_treetops",
    "silva2016_chm",
]


def dalponte2016(
    *,
    chm: NDArray[np.float64],
    seeds: NDArray[np.int32],
    th_tree: float = 2.0,
    th_seed: float = 0.45,
    th_cr: float = 0.55,
    max_cr: float = 10.0,
) -> NDArray[np.int32]:
    """Region-growing tree-crown segmentation (Dalponte & Coomes 2016).

    Translated 1:1 from ``lidR/src/C_dalponte2016.cpp``. Border pixels
    cannot be source pixels for expansion; they may receive growth as
    neighbour destinations only.

    Parameters
    ----------
    chm : (H, W) float64 ndarray, C-contiguous
        Canopy height model in meters.
    seeds : (H, W) int32 ndarray, C-contiguous
        Tree-id seeds; non-zero entries are seed pixels, zero is unassigned.
    th_tree : float, default 2.0
        Pixel height must be strictly greater than this value to be eligible
        to grow into.
    th_seed : float in [0, 1], default 0.45
        Neighbour height must be strictly greater than
        th_seed * seed_height to expand.
    th_cr : float in [0, 1], default 0.55
        Neighbour height must be strictly greater than
        th_cr * mean_crown_height to expand.
    max_cr : float, default 10.0
        Strict Chebyshev distance limit (|Δrow|, |Δcol|) from seed.
    """
    return _core.dalponte2016(
        chm=chm,
        seeds=seeds,
        th_tree=th_tree,
        th_seed=th_seed,
        th_cr=th_cr,
        max_cr=max_cr,
    )


def silva2016(
    *,
    xyz: NDArray[np.float64],
    treetops: NDArray[np.float64],
    max_cr_factor: float = 0.6,
    exclusion: float = 0.3,
) -> NDArray[np.int32]:
    """Voronoi-tesselation tree segmentation (Silva et al. 2016).

    Adapted from ``lidR/R/algorithm-its.R::silva2016`` (line 203). lidR
    operates on a CHM (raster cells); this Python port operates directly on
    the point cloud — each input point is assigned to its nearest treetop
    in the xy plane, then filtered by per-tree height and distance.

    Parameters
    ----------
    xyz : (N, 3) float64 ndarray
        Point cloud (x, y, z).
    treetops : (M, 3) float64 ndarray
        Treetop coordinates. Only the xy columns are consulted; z is
        accepted but unused (matches lidR which uses sf::st_coordinates,
        which exposes only xy).
    max_cr_factor : float, default 0.6
        Maximum crown radius as a fraction of tree height. A point at
        distance d from its assigned treetop is kept iff d ≤ max_cr_factor
        * hmax, where hmax is the tallest point assigned to that tree.
    exclusion : float in (0, 1), default 0.3
        Lower height cutoff as a fraction of hmax: points with z < exclusion
        * hmax are dropped from their tree.

    Returns
    -------
    (N,) int32 ndarray
        Per-point tree id in 1..M (1-based, matching lidR's treeID
        convention); 0 marks unassigned points.

    Notes
    -----
    Port adaptation: lidR computes ``hmax`` as ``max(Z)`` over CHM raster
    cells assigned to a tree. This port operates on the point cloud
    directly, so ``hmax`` is ``max(z)`` over the input points assigned to
    a tree. Outputs may therefore differ slightly from lidR running on a
    CHM derived from the same scene.
    """
    # Spec §3 mandates dtype/shape/contig checks at the bindings entry. For
    # this pure-Python algorithm, the public function entry plays that role,
    # so checks mirror the strictness of nanobind's `.noconvert()` over in
    # `dalponte2016`. Python convention: TypeError for wrong dtype, ValueError
    # for shape / range / contig.
    if not isinstance(xyz, np.ndarray):
        raise TypeError("silva2016: xyz must be a numpy ndarray")
    if xyz.dtype != np.float64:
        raise TypeError("silva2016: xyz must be float64")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("silva2016: xyz must have shape (N, 3)")
    if not xyz.flags.c_contiguous:
        raise ValueError("silva2016: xyz must be C-contiguous")

    if not isinstance(treetops, np.ndarray):
        raise TypeError("silva2016: treetops must be a numpy ndarray")
    if treetops.dtype != np.float64:
        raise TypeError("silva2016: treetops must be float64")
    if treetops.ndim != 2 or treetops.shape[1] != 3:
        raise ValueError("silva2016: treetops must have shape (M, 3)")
    if not treetops.flags.c_contiguous:
        raise ValueError("silva2016: treetops must be C-contiguous")

    if not (max_cr_factor > 0.0):
        raise ValueError("silva2016: max_cr_factor must be > 0")
    if not (0.0 < exclusion < 1.0):
        raise ValueError("silva2016: exclusion must be in (0, 1)")

    n = xyz.shape[0]
    m = treetops.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n,), dtype=np.int32)

    # Voronoi tesselation = k=1 nearest neighbour in xy.
    tree = cKDTree(treetops[:, :2])
    d, idx = tree.query(xyz[:, :2], k=1)
    idx = np.asarray(idx, dtype=np.int64)
    d = np.asarray(d, dtype=np.float64)

    # hmax[id] := max(z) over points assigned to tree `id`.
    z = xyz[:, 2]
    hmax = np.full((m,), -np.inf, dtype=np.float64)
    np.maximum.at(hmax, idx, z)

    keep = (z >= exclusion * hmax[idx]) & (d <= max_cr_factor * hmax[idx])

    out = np.zeros((n,), dtype=np.int32)
    out[keep] = (idx[keep] + 1).astype(np.int32)
    return out


def li2012(
    *,
    xyz: NDArray[np.float64],
    dt1: float = 1.5,
    dt2: float = 2.0,
    R: float = 2.0,
    Zu: float = 15.0,
    hmin: float = 2.0,
    speed_up: float = 10.0,
) -> NDArray[np.int32]:
    """Tree segmentation (Li, Guo, Jakubowski & Kelly 2012).

    1:1 translation of ``lidR/src/LAS.cpp::segment_trees`` (line 1113). The
    in-tree distance pre-pass uses lidR's local-maximum filter with a
    circular window of diameter ``R``. First-version is O(N²) (no KD-tree
    acceleration); a future perf PR may speed up the inner ``sqdistance``
    sweeps.

    Parameters
    ----------
    xyz : (N, 3) float64 ndarray, C-contiguous
        Point cloud.
    dt1 : float, default 1.5
        Distance threshold (xy) used while ``z ≤ Zu``.
    dt2 : float, default 2.0
        Distance threshold (xy) used while ``z > Zu``.
    R : float, default 2.0
        LMF window diameter for the local-max pre-pass. ``R = 0`` skips it
        (every point counts as a local max — fastest, less selective).
    Zu : float, default 15.0
        Z cutoff between dt1 (low) and dt2 (high) regimes.
    hmin : float, default 2.0
        Stop when the highest remaining point falls below this. Mapped to
        lidR's ``th_tree`` parameter.
    speed_up : float, default 10.0
        Distance cap from current treetop; points farther than this are
        auto-rejected without applying Li 2012 rules. Mapped to lidR's
        ``radius`` parameter.
    """
    return _core.li2012(
        xyz=xyz, dt1=dt1, dt2=dt2, Zu=Zu, R=R, hmin=hmin, speed_up=speed_up,
    )


def lmf_points(
    *,
    xyz: NDArray[np.float64],
    ws: Union[float, NDArray[np.float64], Callable[[float], float]],
    hmin: float = 2.0,
    shape: str = "circular",
) -> NDArray[np.bool_]:
    """Local-maximum filter on a point cloud.

    1:1 translation of ``lidR/src/LAS.cpp::filter_local_maxima`` (line 399),
    sequential. The R-side wrapper (``lidR/R/algorithm-itd.R::lmf``)
    materialises a callable ``ws`` into a per-point array before calling
    C++; we mirror that expansion here so the C++ entry always takes a
    length-N float64 array (spec §3).

    Parameters
    ----------
    xyz : (N, 3) float64 ndarray, C-contiguous
        Point cloud.
    ws : float | (N,) float64 ndarray | callable
        Window size (full diameter for ``shape='circular'``, full side
        length for ``shape='square'``). A scalar applies to all points; an
        array gives per-point windows; a callable is called once per point
        with that point's z value and must return a positive float.
    hmin : float, default 2.0
        Points with ``z < hmin`` cannot be local maxima.
    shape : {'circular', 'square'}
        Window shape.

    Returns
    -------
    (N,) bool ndarray
        ``True`` at indices where the point is a local maximum within its
        window.
    """
    if not isinstance(xyz, np.ndarray):
        raise TypeError("lmf_points: xyz must be a numpy ndarray")
    if xyz.dtype != np.float64:
        raise TypeError("lmf_points: xyz must be float64")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("lmf_points: xyz must have shape (N, 3)")
    if not xyz.flags.c_contiguous:
        raise ValueError("lmf_points: xyz must be C-contiguous")

    n = xyz.shape[0]

    # Expand ws → (N,) float64. Track is_uniform so the C++ side can apply
    # lidR's cascading-NLM optimisation only when valid.
    is_uniform = False
    if callable(ws):
        ws_values: list[float] = []
        for z in xyz[:, 2]:
            value = np.asarray(ws(float(z)), dtype=np.float64)
            if value.shape != ():
                raise ValueError(
                    "lmf_points: ws callable must return a scalar for each z, "
                    f"got shape {value.shape}"
                )
            ws_values.append(float(value))
        ws_arr = np.asarray(ws_values, dtype=np.float64)
    elif np.isscalar(ws):
        ws_arr = np.full((n,), float(ws), dtype=np.float64)
        is_uniform = True
    else:
        ws_arr = np.asarray(ws)
        if ws_arr.dtype != np.float64:
            raise TypeError("lmf_points: ws array must be float64")
        if ws_arr.shape != (n,):
            raise ValueError(
                f"lmf_points: ws array must have shape ({n},), "
                f"got {ws_arr.shape}"
            )
    if not ws_arr.flags.c_contiguous:
        ws_arr = np.ascontiguousarray(ws_arr)

    if shape not in ("circular", "square"):
        raise ValueError("lmf_points: shape must be 'circular' or 'square'")

    return _core.lmf_points(
        xyz=xyz, ws=ws_arr, hmin=hmin, shape=shape, is_uniform=is_uniform,
    )


def lmf_chm(
    *,
    chm: NDArray[np.float64],
    ws: float,
    hmin: float = 2.0,
    shape: str = "circular",
) -> NDArray[np.int32]:
    """Local-maximum filter on a CHM raster.

    Equivalent to running ``lmf_points`` on the point cloud you'd get by
    dropping each non-NaN cell at its grid centre. Implemented as a direct
    raster scan (no kdtree build, no point-cloud materialisation) — the
    only intentional optimisation over a literal lidR mirror.

    Parameters
    ----------
    chm : (H, W) float64 ndarray, C-contiguous
        Canopy height model. NaN cells are treated as nodata: never a
        local max, never counted as a neighbour.
    ws : float
        Window size in **pixel** units. For a CHM with non-unit cell size,
        convert world units → pixel units before calling. Fractional ws is
        permitted; the integer half-extent of the search box is
        ``floor(ws/2)``, matching lidR's Rectangle on a unit-cell raster
        (the EPSILON slack from lidR's ``Shapes.h`` cannot include any
        extra integer cell, so floor is exact).
    hmin : float, default 2.0
        Cells with ``z < hmin`` cannot be local maxima.
    shape : {'circular', 'square'}
        Window shape.

    Returns
    -------
    (K, 2) int32 ndarray
        Row/column indices of all detected local maxima, in row-major scan
        order.
    """
    if shape not in ("circular", "square"):
        raise ValueError("lmf_chm: shape must be 'circular' or 'square'")
    return _core.lmf_chm(chm=chm, ws=float(ws), hmin=hmin, shape=shape)


def chm_smooth(
    *,
    xyz: NDArray[np.float64],
    size: float = 3.0,
    method: str = "average",
    shape: str = "circular",
    sigma: float | None = None,
) -> NDArray[np.float64]:
    """Point-cloud height smoothing (lidR ``smooth_height``).

    1:1 translation of ``lidR/src/LAS.cpp::z_smooth`` (line 112). For each
    input point i, neighbours within a circular disc (radius ``size/2``)
    or square box (half-side ``size/2``) of (x_i, y_i) — including i
    itself — contribute to a weighted mean of z; the mean replaces z_i in
    the output.

    Parameters
    ----------
    xyz : (N, 3) float64 ndarray, C-contiguous
        Point cloud.
    size : float, default 3.0
        Full window size (full diameter for ``shape='circular'``, full
        side length for ``shape='square'``).
    method : {'average', 'gaussian'}
        ``'average'`` weights every neighbour equally; ``'gaussian'``
        weights by ``(1 / (2σ²π)) · exp(-d² / (2σ²))`` where d is the
        xy-plane distance to the query point and σ is ``sigma``.
    shape : {'circular', 'square'}
        Window shape.
    sigma : float | None, default None → ``size / 6``
        Standard deviation of the Gaussian kernel (used only when
        ``method == 'gaussian'``). The default mirrors lidR's R-side
        wrapper (``smooth_height.R:34``: ``sigma = size/6``); spec §3
        omits sigma but a Gaussian smoother needs it, so we expose it
        with the lidR R default — see findings.md.

    Returns
    -------
    (N,) float64 ndarray
        Smoothed z column. ``out[i]`` is the weighted mean of the z
        values of all neighbours of point i (point i included).

    Notes
    -----
    Per lidR's algorithm, NaN inputs are not guarded — a NaN neighbour z
    propagates a NaN into the output. Filter NaNs upstream if needed.

    The R-side wrapper ``smooth_height.R:45`` has a known typo
    (``if (method == "circle")`` instead of ``if (shape == "circle")``)
    that effectively forces ``shape='circular'`` in lidR regardless of
    the user's argument. This port translates the C function directly,
    so ``shape`` is honoured.
    """
    if not isinstance(xyz, np.ndarray):
        raise TypeError("chm_smooth: xyz must be a numpy ndarray")
    if xyz.dtype != np.float64:
        raise TypeError("chm_smooth: xyz must be float64")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("chm_smooth: xyz must have shape (N, 3)")
    if not xyz.flags.c_contiguous:
        raise ValueError("chm_smooth: xyz must be C-contiguous")
    if not (size > 0.0):
        raise ValueError("chm_smooth: size must be > 0")
    if method not in ("average", "gaussian"):
        raise ValueError("chm_smooth: method must be 'average' or 'gaussian'")
    if shape not in ("circular", "square"):
        raise ValueError("chm_smooth: shape must be 'circular' or 'square'")

    if sigma is None:
        sigma = size / 6.0
    if not (sigma > 0.0):
        raise ValueError("chm_smooth: sigma must be > 0")

    return _core.chm_smooth(
        xyz=xyz, size=float(size), method=method, shape=shape,
        sigma=float(sigma),
    )


def watershed(
    *,
    chm: NDArray[np.float64],
    th_tree: float = 2.0,
    tol: float = 1.0,
    ext: int = 1,
) -> NDArray[np.int32]:
    """Watershed-based crown segmentation on a CHM (EBImage parity).

    1:1 port of lidR ``R/algorithm-its.R::watershed`` (L328-377), which
    delegates to ``EBImage::watershed``. Phase 4.5 (2026-05-13) replaced
    the previous skimage approximation with a direct C++ port of
    ``EBImage src/watershed.cpp`` (see :func:`pylidar._core.watershed_ext`
    + ``src/core/its/watershed_ext.cpp`` PORT NOTE).

    Pipeline (mirrors lidR R wrapper):

    1. ``Canopy[Canopy < th_tree | is.na(Canopy)] <- 0`` (mask + cleanup).
    2. ``Crowns <- _core.watershed_ext(Canopy, tolerance=tol, ext=ext)``
       — height-descending priority-queue flood with EBImage's
       multi-neighbour merge / tolerance semantics.
    3. ``Crowns[mask] <- 0`` (explicit zero-fill on background; lidR
       writes ``NA_integer_`` here, we render NA as 0 to match the
       Python contract used by :func:`pylidar.merge.merge_raster_labels`).

    Parameters
    ----------
    chm : (H, W) float64 ndarray
        Canopy height model in meters. NaN cells are treated as below
        ``th_tree`` (no tree).
    th_tree : float, default 2.0
        Cells with ``chm < th_tree`` (or NaN) become 0 in the output and
        are excluded from the flood.
    tol : float, default 1.0
        Minimum height drop from a regional maximum to seed a distinct
        basin; mapped 1:1 to EBImage's ``tolerance``.
    ext : int, default 1
        Neighbourhood half-side in pixels. ``ext=1`` → 3×3 (8-connected);
        ``ext=2`` → 5×5; etc. Mapped 1:1 to EBImage's ``ext``.

    Returns
    -------
    (H, W) int32 ndarray
        Tree id per pixel: 0 = below ``th_tree`` / NaN / unassigned,
        positive integers = tree ids contiguous from 1.

    Notes
    -----
    Determinism on plateaus: EBImage relies on R's ``rsort_with_index``
    which is not stable, so EBImage outputs are non-deterministic on
    flat regions. This port uses ``std::stable_sort`` with a row-major
    tie-breaker, so our output is deterministic and reproducible — but
    on plateaus may differ from EBImage's specific (non-deterministic)
    choice. On non-plateau inputs the two should agree bit-exactly
    (R-side byte-level fixture comparison was deferred in Phase 4.5;
    no R env in build host).

    The previous skimage-based approximation (``h_maxima`` + flood on
    negated CHM) is no longer used at runtime. It remains useful as a
    sanity-check baseline if the C++ kernel ever gets refactored.
    """
    if not isinstance(chm, np.ndarray):
        raise TypeError("watershed: chm must be a numpy ndarray")
    if chm.dtype != np.float64:
        raise TypeError("watershed: chm must be float64")
    if chm.ndim != 2:
        raise ValueError("watershed: chm must have shape (H, W)")
    if not chm.flags.c_contiguous:
        raise ValueError("watershed: chm must be C-contiguous")
    if not (tol >= 0.0):
        raise ValueError("watershed: tol must be >= 0")
    if not isinstance(ext, (int, np.integer)) or ext < 1:
        raise ValueError("watershed: ext must be an integer >= 1")

    h, w = chm.shape

    # Mirror lidR's R wrapper exactly: NaN → 0; below-threshold → 0.
    # The C++ kernel skips cells with src <= 0, so masked cells receive
    # label 0 automatically; we don't re-zero after the call.
    canopy = np.where(np.isnan(chm) | (chm < th_tree), 0.0, chm)
    canopy = np.ascontiguousarray(canopy, dtype=np.float64)

    return _core.watershed_ext(chm=canopy, tolerance=float(tol), ext=int(ext))


# ─────────────────────────────────────────────── Phase 4 high-level wrappers
#
# PORT NOTE (Phase 4, 2026-05-12)
# -------------------------------
# Adapted from lidR ``R/algorithm-its.R``: dalponte2016 high-level wrapper
# (L59-146), silva2016 high-level wrapper (L203-283). The existing
# :func:`dalponte2016` (a low-level seed-grid primitive) and
# :func:`silva2016` (a point-cloud Voronoi adaptation) are preserved
# unchanged — these new ``_from_treetops`` / ``_chm`` entry points sit on
# top, taking a :class:`pylidar.locate_trees.Treetops` and returning an
# (H, W) raster of tree IDs whose dtype matches ``treetops.tree_id.dtype``.
#
# Cell-collision policy for seed rasterization mirrors lidR: when two
# treetops fall in the same cell, the **last one wins** (lidR uses
# ``stars::st_rasterize`` with default last-write semantics). For the
# common case of one treetop per cell this is moot.
#
# Output dtype mirrors lidR's ``storage.mode(val) <- storage.mode(treetops[[ID]])``
# (``algorithm-its.R:139``): int32 raster for incremental tree IDs, float64
# raster for gpstime/bitmerge. Unassigned cells are 0 for int32, 0.0 for
# float64 (NOT NaN — lidR uses ``NA_integer_`` which we render as 0 to
# match the existing pylidar output convention; downstream merge code
# treats 0 as "no tree").

def dalponte2016_from_treetops(
    *,
    chm: NDArray[np.float64],
    layout,  # pylidar.raster.RasterLayout — kept untyped to avoid circular import
    treetops,  # pylidar.locate_trees.Treetops
    th_tree: float = 2.0,
    th_seed: float = 0.45,
    th_cr: float = 0.55,
    max_cr: float = 10.0,
) -> NDArray:
    """Dalponte 2016 region-growing seeded by a Treetops table.

    Mirrors lidR ``algorithm-its.R::dalponte2016`` (L118-140). Builds a
    seed grid by rasterizing ``treetops.x/y`` onto ``layout`` (sequential
    1..K seed values, last-wins for cell collisions), runs the C++
    ``_core.dalponte2016`` kernel, then remaps each cell's sequential ID
    back to ``treetops.tree_id`` so the output preserves the user's tree
    ID space (incremental → int32, gpstime/bitmerge → float64).

    Parameters
    ----------
    chm : (H, W) float64 ndarray
        Canopy height model. NaN cells are treated as below ``th_tree``
        for the kernel (substituted with ``-inf`` since the C kernel
        compares against a height threshold, not a NaN-aware comparator).
    layout : RasterLayout
        Spatial extent for the CHM; used to map ``(x, y) → (row, col)``
        for the seed grid.
    treetops : Treetops
        Detected treetops with the user's chosen ID space.
    th_tree, th_seed, th_cr, max_cr : float
        Forwarded to :func:`dalponte2016`.

    Returns
    -------
    (H, W) ndarray
        Tree-id raster, dtype matches ``treetops.tree_id.dtype``.
        Unassigned cells are 0 (or 0.0 for float64).
    """
    # Local import to avoid circular module-load (locate_trees imports
    # segmentation.lmf_points; segmentation.dalponte2016_from_treetops needs
    # a Treetops only at call time, not at import time).
    from .locate_trees import Treetops as _Treetops

    if not isinstance(treetops, _Treetops):
        raise TypeError("treetops must be a pylidar.locate_trees.Treetops")
    if not isinstance(chm, np.ndarray):
        raise TypeError("dalponte2016_from_treetops: chm must be a numpy ndarray")
    if chm.dtype != np.float64:
        raise TypeError("dalponte2016_from_treetops: chm must be float64")
    if chm.ndim != 2:
        raise ValueError("dalponte2016_from_treetops: chm must be (H, W)")
    if not chm.flags.c_contiguous:
        raise ValueError("dalponte2016_from_treetops: chm must be C-contiguous")
    if chm.shape != layout.shape:
        raise ValueError(
            f"dalponte2016_from_treetops: chm.shape {chm.shape} must match "
            f"layout.shape {layout.shape}"
        )

    h, w = chm.shape
    seeds = np.zeros((h, w), dtype=np.int32)

    if treetops.n > 0:
        rows, cols = layout.cell_xy_to_rowcol(treetops.x, treetops.y)
        in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if in_bounds.any():
            r = rows[in_bounds].astype(np.int64)
            c = cols[in_bounds].astype(np.int64)
            # Sequential 1..K_in, in original treetops order. lidR uses
            # `1:nrow(treetops)` for the SEQUENTIALIDTREE column and stars
            # rasterization is last-wins for cell collisions; numpy advanced
            # assignment matches: later writes overwrite earlier ones.
            seq = (np.flatnonzero(in_bounds) + 1).astype(np.int32)
            seeds[r, c] = seq

    # _core.dalponte2016 expects a finite CHM; substitute -inf for NaN so
    # the comparator never silently evaluates `nan > th_tree` as False
    # (which is the right behavior, but we don't want to leave it implicit).
    chm_clean = np.where(np.isnan(chm), -np.inf, chm)
    chm_clean = np.ascontiguousarray(chm_clean, dtype=np.float64)

    crowns = _core.dalponte2016(
        chm=chm_clean,
        seeds=seeds,
        th_tree=th_tree,
        th_seed=th_seed,
        th_cr=th_cr,
        max_cr=max_cr,
    )

    # Remap sequential IDs (1..K) → treetops.tree_id values. lidR:
    # `Crowns[] <- treetops[[ID]][Crowns]` — cells with crowns==0 stay 0.
    out_dtype = treetops.tree_id.dtype
    out = np.zeros((h, w), dtype=out_dtype)
    nonzero = crowns > 0
    if nonzero.any():
        # Only ids assigned in `seeds` will appear in `crowns`; map each via
        # the original treetops.tree_id array (1-indexed → 0-indexed lookup).
        idx = crowns[nonzero] - 1
        # idx may legally exceed treetops.n - 1 if the kernel produces an
        # ID we never seeded; defensive cap (should not happen).
        if idx.max() >= treetops.n:
            raise RuntimeError(
                f"_core.dalponte2016 produced id {int(idx.max()) + 1} > "
                f"treetops.n {treetops.n}"
            )
        out[nonzero] = treetops.tree_id[idx]
    return out


def silva2016_chm(
    *,
    chm: NDArray[np.float64],
    layout,  # pylidar.raster.RasterLayout
    treetops,  # pylidar.locate_trees.Treetops
    max_cr_factor: float = 0.6,
    exclusion: float = 0.3,
) -> NDArray:
    """Silva 2016 Voronoi+hmax segmentation on a CHM.

    1:1 port of lidR ``algorithm-its.R::silva2016`` (L257-277). For every
    non-NaN CHM cell:

    1. Find the nearest treetop in xy (``cKDTree.query(k=1)``); the
       resulting per-cell tree assignment is the Voronoi partition.
    2. Compute ``hmax`` per tree as the maximum CHM cell value among cells
       assigned to that tree (NB: this is **CHM-cell hmax**, not point-cloud
       hmax — the existing :func:`silva2016` adaptation uses point-cloud z).
    3. Keep cells satisfying ``z >= exclusion * hmax`` and
       ``d <= max_cr_factor * hmax``. All other cells become 0
       (unassigned).
    4. Map the per-cell tree index back to ``treetops.tree_id``.

    Parameters
    ----------
    chm : (H, W) float64 ndarray
    layout : RasterLayout
    treetops : Treetops
    max_cr_factor : float, default 0.6
    exclusion : float in (0, 1), default 0.3

    Returns
    -------
    (H, W) ndarray
        Tree-id raster, dtype matches ``treetops.tree_id.dtype``.
    """
    from .locate_trees import Treetops as _Treetops

    if not isinstance(treetops, _Treetops):
        raise TypeError("treetops must be a pylidar.locate_trees.Treetops")
    if not isinstance(chm, np.ndarray):
        raise TypeError("silva2016_chm: chm must be a numpy ndarray")
    if chm.dtype != np.float64:
        raise TypeError("silva2016_chm: chm must be float64")
    if chm.ndim != 2:
        raise ValueError("silva2016_chm: chm must be (H, W)")
    if not chm.flags.c_contiguous:
        raise ValueError("silva2016_chm: chm must be C-contiguous")
    if chm.shape != layout.shape:
        raise ValueError(
            f"silva2016_chm: chm.shape {chm.shape} must match layout.shape "
            f"{layout.shape}"
        )
    if not (max_cr_factor > 0.0):
        raise ValueError("silva2016_chm: max_cr_factor must be > 0")
    if not (0.0 < exclusion < 1.0):
        raise ValueError("silva2016_chm: exclusion must be in (0, 1)")

    out_dtype = treetops.tree_id.dtype
    h, w = chm.shape
    out = np.zeros((h, w), dtype=out_dtype)

    if treetops.n == 0:
        return out

    finite_mask = ~np.isnan(chm)
    if not finite_mask.any():
        return out

    rows_idx, cols_idx = np.nonzero(finite_mask)
    cell_x, cell_y = layout.rowcol_to_cell_xy(rows_idx, cols_idx)
    cell_z = chm[rows_idx, cols_idx]

    treetops_xy = np.column_stack((treetops.x, treetops.y))
    tree = cKDTree(treetops_xy)
    dists, nn_idx = tree.query(np.column_stack((cell_x, cell_y)), k=1)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)
    dists = np.asarray(dists, dtype=np.float64)

    # hmax[t] = max(cell_z) over cells assigned to tree t.
    hmax = np.full((treetops.n,), -np.inf, dtype=np.float64)
    np.maximum.at(hmax, nn_idx, cell_z)

    keep = (cell_z >= exclusion * hmax[nn_idx]) & (
        dists <= max_cr_factor * hmax[nn_idx]
    )
    if keep.any():
        out[rows_idx[keep], cols_idx[keep]] = treetops.tree_id[nn_idx[keep]]
    return out
