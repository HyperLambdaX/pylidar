"""Public ITS algorithms for pylidar.

Per spec §3, the user-facing API lives here. C++ algorithms come from
`pylidar._core`; pure-Python algorithms (silva2016, watershed) live in this
module directly. All function arguments are keyword-only.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from . import _core

__all__ = ["dalponte2016", "silva2016", "li2012", "lmf_points", "lmf_chm"]


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
    from scipy.spatial import cKDTree

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
