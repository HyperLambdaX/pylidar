import numpy as np
from numpy.typing import NDArray

__version__: str

def ping() -> str: ...

def dalponte2016(
    *,
    chm: NDArray[np.float64],
    seeds: NDArray[np.int32],
    th_tree: float = 2.0,
    th_seed: float = 0.45,
    th_cr: float = 0.55,
    max_cr: float = 10.0,
) -> NDArray[np.int32]: ...

def li2012(
    *,
    xyz: NDArray[np.float64],
    dt1: float = 1.5,
    dt2: float = 2.0,
    Zu: float = 15.0,
    R: float = 2.0,
    hmin: float = 2.0,
    speed_up: float = 10.0,
) -> NDArray[np.int32]: ...

def lmf_points(
    *,
    xyz: NDArray[np.float64],
    ws: NDArray[np.float64],
    hmin: float = 2.0,
    shape: str = "circular",
    is_uniform: bool = False,
) -> NDArray[np.bool_]: ...

def lmf_chm(
    *,
    chm: NDArray[np.float64],
    ws: float,
    hmin: float = 2.0,
    shape: str = "circular",
) -> NDArray[np.int32]: ...

def watershed_ext(
    *,
    chm: NDArray[np.float64],
    tolerance: float = 1.0,
    ext: int = 1,
) -> NDArray[np.int32]: ...

def chm_smooth(
    *,
    xyz: NDArray[np.float64],
    size: float,
    method: str,
    shape: str,
    sigma: float,
) -> NDArray[np.float64]: ...
# Internal entry; no defaults. User-facing defaults live in
# `pylidar.segmentation.chm_smooth`.
