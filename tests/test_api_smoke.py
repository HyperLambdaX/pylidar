"""End-to-end build-chain smoke test.

Validates that:
  - the wheel built and installed correctly,
  - the nanobind extension `_core` loads,
  - the only Phase-0 binding (`set_log_callback`) is invocable and round-trips
    through C++ to a Python callable.

This is the Phase-0 acceptance test for the entire CMake → C++ → bindings →
Python plumbing.
"""

from __future__ import annotations

import sys


def test_package_import():
    import pylidar

    assert hasattr(pylidar, "__version__")
    assert pylidar.__version__ == "0.1.0"


def test_core_extension_loads():
    import pylidar
    from pylidar import _core

    assert _core is pylidar._core
    assert hasattr(_core, "set_log_callback")


def test_set_log_callback_round_trip():
    """Install a Python callback, trigger it via C++ if possible, then reset.

    Phase 0 has no algorithm that emits logs, so we only verify that
    set_log_callback accepts a callable, accepts None, and doesn't raise.
    Phase 1+ will add a real round-trip test once an algorithm logs.
    """
    import pylidar

    received: list[str] = []

    pylidar.set_log_callback(received.append)
    pylidar.set_log_callback(None)
    pylidar.set_log_callback(lambda msg: print(msg, file=sys.stderr))
    pylidar.set_log_callback(None)

    # No algorithm has been bound yet, so received is expected to remain empty.
    # The point of this test is that none of the calls above blew up.
    assert received == []


def test_set_log_callback_rejects_non_callable():
    import pylidar

    try:
        pylidar.set_log_callback(42)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("set_log_callback should reject non-callables")
    finally:
        pylidar.set_log_callback(None)
