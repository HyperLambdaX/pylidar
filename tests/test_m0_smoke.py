"""M0 smoke test: package + native module load and basic call works.

This is the only test at M0; per spec §4 the tests/ tree (conftest, fixtures,
per-algorithm test modules) lands incrementally starting M1.
"""

import pylidar
import pylidar._core as _core


def test_package_version_exposed():
    assert pylidar.__version__ == "0.1.0"


def test_native_module_version_exposed():
    assert _core.__version__ == "0.1.0"


def test_native_ping_returns_pong():
    assert _core.ping() == "pong"
