"""Ground classification algorithms.

This module currently exposes the Progressive Morphological Filter (PMF)
ported from lidR. The public API is array-first: classification returns a
boolean mask where ``True`` means ground.

PORT NOTE
---------
The C++ PMF kernel intentionally preserves lidR's ``z_open`` implementation,
including its all-negative-window quirk from initializing pass-2 maxima with
``std::numeric_limits<double>::min()``. The public Python PMF path applies a
constant positive Z shift before calling that kernel when input heights are not
strictly positive. Morphological opening and the PMF threshold test are
translation invariant in Z, so the shift preserves intended PMF behaviour while
preventing negative absolute elevations from being silently classified as all
ground.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray

from . import _core

__all__ = [
    "CsfAlgorithm",
    "GroundAlgorithm",
    "PmfAlgorithm",
    "classify_ground",
    "csf",
    "pmf",
    "util_makeZhangParam",
]


@dataclass(frozen=True)
class GroundAlgorithm:
    name: str


@dataclass(frozen=True)
class PmfAlgorithm(GroundAlgorithm):
    ws: NDArray[np.float64]
    th: NDArray[np.float64]


@dataclass(frozen=True)
class CsfAlgorithm(GroundAlgorithm):
    slope_smooth: bool
    class_threshold: float
    cloth_resolution: float
    rigidness: int
    iterations: int
    time_step: float


def _pmf_xyz_for_core(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    z_min = float(xyz[:, 2].min()) if xyz.shape[0] else 0.0
    if z_min > 0.0:
        return xyz
    out = xyz.copy()
    out[:, 2] += -z_min + 1.0
    return np.ascontiguousarray(out, dtype=np.float64)


def util_makeZhangParam(
    *,
    b: float = 2.0,
    dh0: float = 0.5,
    dhmax: float = 3.0,
    s: float = 1.0,
    max_ws: float = 20.0,
    exp: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Zhang PMF window/threshold sequences.

    Mirrors lidR ``R/algorithm-gnd.R::util_makeZhangParam``.
    """
    b = float(b)
    dh0 = float(dh0)
    dhmax = float(dhmax)
    s = float(s)
    max_ws = float(max_ws)

    if not isinstance(exp, (bool, np.bool_)):
        raise TypeError("exp should be logical")
    exp = bool(exp)
    if exp and b <= 1.0:
        raise ValueError(
            "b cannot be less than 1 with an exponentially growing window"
        )
    if dh0 >= dhmax:
        raise ValueError("dh0 greater than dhmax")
    if max_ws < 3.0:
        raise ValueError("Minimum windows size is 3. max_ws must be greater than 3")
    if (not exp) and b < 1.0:
        warnings.warn(
            "Due to an incoherence in the original paper when b < 1, the "
            "sequences of windows size cannot be computed for a linear "
            "increase. The internal routine uses the fact that the increment "
            "is constant to bypass this issue.",
            RuntimeWarning,
            stacklevel=2,
        )

    dhtk: list[float] = []
    wk: list[float] = []
    k = 0
    ws = 0.0
    th = 0.0
    c = 1.0

    while ws <= max_ws:
        if exp:
            ws = (2.0 * (b ** k)) + 1.0
        else:
            ws = 2.0 * (k + 1) * b + 1.0

        if ws <= 3.0:
            th = dh0
        elif exp:
            th = s * (ws - wk[k - 1]) * c + dh0
        else:
            th = s * 2.0 * b * c + dh0

        if th > dhmax:
            th = dhmax

        if ws <= max_ws:
            wk.append(float(ws))
            dhtk.append(float(th))

        k += 1

    return (
        np.ascontiguousarray(wk, dtype=np.float64),
        np.ascontiguousarray(dhtk, dtype=np.float64),
    )


def pmf(
    *,
    ws: NDArray[np.float64],
    th: NDArray[np.float64],
) -> PmfAlgorithm:
    """Create a PMF ground algorithm descriptor."""
    ws_arr = np.ascontiguousarray(ws, dtype=np.float64)
    th_arr = np.ascontiguousarray(th, dtype=np.float64)
    if ws_arr.ndim != 1:
        raise ValueError("pmf: ws must be 1-D")
    if th_arr.ndim != 1:
        raise ValueError("pmf: th must be 1-D")
    if ws_arr.shape[0] != th_arr.shape[0]:
        raise ValueError("pmf: ws and th length mismatch")
    if not np.all(np.isfinite(ws_arr)) or np.any(ws_arr <= 0.0):
        raise ValueError("pmf: ws must be finite and > 0")
    if not np.all(np.isfinite(th_arr)) or np.any(th_arr <= 0.0):
        raise ValueError("pmf: th must be finite and > 0")
    return PmfAlgorithm(name="pmf", ws=ws_arr, th=th_arr)


def csf(
    *,
    slope_smooth: bool = False,
    class_threshold: float = 0.5,
    cloth_resolution: float = 0.5,
    rigidness: int = 1,
    iterations: int = 500,
    time_step: float = 0.65,
) -> CsfAlgorithm:
    """Create a CSF ground algorithm descriptor.

    Defaults mirror lidR ``csf()`` / RCSF: ``sloop_smooth = FALSE``,
    ``class_threshold = 0.5``, ``cloth_resolution = 0.5``,
    ``rigidness = 1``, ``iterations = 500``, ``time_step = 0.65``.
    """
    if not isinstance(slope_smooth, (bool, np.bool_)):
        raise TypeError("csf: slope_smooth must be bool")
    class_threshold = float(class_threshold)
    cloth_resolution = float(cloth_resolution)
    time_step = float(time_step)
    rigidness = int(rigidness)
    iterations = int(iterations)
    if not np.isfinite(class_threshold) or class_threshold <= 0.0:
        raise ValueError("csf: class_threshold must be finite and > 0")
    if not np.isfinite(cloth_resolution) or cloth_resolution <= 0.0:
        raise ValueError("csf: cloth_resolution must be finite and > 0")
    if rigidness < 1 or rigidness > 3:
        raise ValueError("csf: rigidness must be in [1, 3]")
    if iterations <= 0:
        raise ValueError("csf: iterations must be > 0")
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("csf: time_step must be finite and > 0")
    return CsfAlgorithm(
        name="csf",
        slope_smooth=bool(slope_smooth),
        class_threshold=class_threshold,
        cloth_resolution=cloth_resolution,
        rigidness=rigidness,
        iterations=iterations,
        time_step=time_step,
    )


def classify_ground(
    *,
    xyz: NDArray[np.float64],
    algorithm: GroundAlgorithm,
    candidate_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Classify ground points and return a boolean mask.

    ``candidate_mask=None`` means every point is a ground candidate. This is
    the explicit pylidar equivalent of lidR's last-return prefilter being
    handled by the LAS layer.
    """
    xyz_arr = np.ascontiguousarray(xyz, dtype=np.float64)
    if xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3:
        raise ValueError("classify_ground: xyz must have shape (N, 3)")
    if not np.all(np.isfinite(xyz_arr)):
        raise ValueError("classify_ground: xyz must be finite")
    if candidate_mask is None:
        candidate = np.ones(xyz_arr.shape[0], dtype=np.bool_)
    else:
        candidate = np.ascontiguousarray(candidate_mask, dtype=np.bool_)
        if candidate.ndim != 1:
            raise ValueError("classify_ground: candidate_mask must be 1-D")
        if candidate.shape[0] != xyz_arr.shape[0]:
            raise ValueError(
                "classify_ground: candidate_mask length must equal xyz row count"
            )

    if isinstance(algorithm, PmfAlgorithm):
        xyz_core = _pmf_xyz_for_core(xyz_arr)
        return _core.pmf_ground(
            xyz=xyz_core,
            ws=algorithm.ws,
            th=algorithm.th,
            candidate_mask=candidate,
        )
    if isinstance(algorithm, CsfAlgorithm):
        return _core.csf_ground(
            xyz=xyz_arr,
            candidate_mask=candidate,
            slope_smooth=algorithm.slope_smooth,
            class_threshold=algorithm.class_threshold,
            cloth_resolution=algorithm.cloth_resolution,
            rigidness=algorithm.rigidness,
            iterations=algorithm.iterations,
            time_step=algorithm.time_step,
        )
    raise TypeError(f"unsupported ground algorithm: {algorithm!r}")
