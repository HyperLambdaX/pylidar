"""Matrix2D semantics tests.

Matrix2D is a C++ type with no Python binding in v0.1 — Python users see
numpy arrays only, with the column-major ↔ row-major transpose handled
inside the bindings layer. So Phase 0 has nothing to assert at the Python
level; the column-major invariant is validated indirectly in Phase 1+ by the
algorithm tests (a row/column mix-up would corrupt CHM outputs).

This file exists to satisfy the Phase-0 acceptance gate (task_plan.md) and
holds a single skipped placeholder. When the first algorithm wrapper exposes
a CHM round-trip, replace the skip with concrete assertions:

  - chm written by Python at (row=2, col=3) reads back from algorithm output
    at (row=2, col=3), proving the in-bindings transpose pairs correctly,
  - bounds checks reject negative indices and out-of-range coordinates.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Matrix2D is C++-only in v0.1; concrete tests added in Phase 1+")
def test_matrix2d_column_major_round_trip():
    pass
