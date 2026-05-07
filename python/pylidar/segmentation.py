"""Public ITS algorithms for pylidar.

Per spec §3, the user-facing API lives here. C++ algorithms come from
`pylidar._core`; pure-Python algorithms (silva2016, watershed) live in this
module directly. All function arguments are keyword-only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _core

__all__ = ["dalponte2016", "silva2016"]


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
