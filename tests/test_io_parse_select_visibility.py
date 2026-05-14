"""Phase 5 audit-fix #4 (2026-05-12): ``parse_select`` is a parser helper
that changed return shape during the Phase 2/3 audit (3-tuple → 4-tuple).
To prevent future churn from being mis-read as a stable API break, it is
no longer in ``pylidar.io.__all__`` — callers using ``from pylidar.io
import *`` won't bind it. The symbol is still accessible by name for
internal use and existing tests."""

from __future__ import annotations

import pylidar.io


def test_parse_select_not_in_all():
    assert "parse_select" not in pylidar.io.__all__


def test_parse_select_still_importable_for_internal_use():
    """Direct access is intentionally preserved."""
    from pylidar.io import parse_select  # noqa: F401
