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
