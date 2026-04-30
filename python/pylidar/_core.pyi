"""Type stubs for the C++ extension `pylidar._core`.

Hand-written; kept in sync with src/bindings/module.cpp. Algorithm function
stubs are added phase-by-phase as their bindings land.
"""

from __future__ import annotations

from typing import Callable, Optional

def set_log_callback(callback: Optional[Callable[[str], None]]) -> None:
    """Install a Python callable to receive log messages from the C++ core.

    Pass ``None`` to disable logging (the default). Exceptions raised by the
    callback are swallowed and routed through :func:`sys.unraisablehook`.
    """
    ...
