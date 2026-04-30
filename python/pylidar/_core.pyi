"""Type stubs for the C++ extension `pylidar._core`.

Hand-written; kept in sync with src/bindings/module.cpp. Algorithm function
stubs are added phase-by-phase as their bindings land.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

def set_log_callback(callback: Optional[Callable[[str], None]]) -> None:
    """Install a Python callable to receive log messages from the C++ core.

    Pass ``None`` to disable logging (the default). Exceptions raised by the
    callback are swallowed and routed through :func:`sys.unraisablehook`.
    """
    ...

def smooth_height(
    xyz: npt.NDArray[np.float64],
    size: float,
    method: int,
    shape: int,
    sigma: float,
) -> npt.NDArray[np.float64]:
    """Internal: smooth point-cloud Z values.

    method: 1=mean, 2=gaussian. shape: 1=square, 2=circular.

    Use :func:`pylidar.smooth_height` instead — it validates inputs and
    accepts the public string-based API.
    """
    ...
