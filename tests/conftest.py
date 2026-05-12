"""Shared test utilities.

Per spec §6, algorithm fixtures live as .npz files under tests/fixtures/.
Each carries ``inputs/*``, ``expected/*``, and ``meta/*`` keys; the
``load_fixture`` pytest fixture returns the union as a flat dict so callers
can unpack with explicit prefixes.

Phase 2 (read_las / point_mask) tests need real LAS files, not .npz arrays.
The ``synthetic_las_path`` fixture builds a tiny in-memory ``laspy.LasData``
and writes it to ``tmp_path``. Returns the path; callers reload via
``read_las`` so the round-trip path is exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import laspy
import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    path = FIXTURE_DIR / f"{name}.npz"
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


@pytest.fixture
def load_fixture() -> Callable[[str], dict]:
    return _load_fixture


def _build_synthetic_las(
    *,
    n: int = 12,
    point_format: int = 3,
    version: str = "1.2",
    classifications: np.ndarray | None = None,
    return_numbers: np.ndarray | None = None,
    number_of_returns: np.ndarray | None = None,
    intensities: np.ndarray | None = None,
) -> laspy.LasData:
    """Build a small ``laspy.LasData`` with deterministic, hand-traceable
    point attributes. Defaults give 12 points on a 4×3 xy lattice with z
    increasing along the row index, ground (class 2) and noise (class 7)
    interleaved, two returns per point so first/last/single semantics fire.
    """
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    header.offsets = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    las = laspy.LasData(header)

    xs = np.tile(np.arange(4, dtype=np.float64), 3)
    ys = np.repeat(np.arange(3, dtype=np.float64), 4)
    zs = np.arange(n, dtype=np.float64) * 0.5  # 0.0, 0.5, 1.0, ..., 5.5

    las.x = xs
    las.y = ys
    las.z = zs

    if classifications is None:
        # alternating ground (2) / noise (7) / vegetation (5)
        classifications = np.array(
            [2, 7, 5, 2, 7, 5, 2, 7, 5, 2, 7, 5], dtype=np.uint8
        )
    las.classification = classifications

    if return_numbers is None:
        # half "first of two", half "last of two"
        return_numbers = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.uint8)
    las.return_number = return_numbers

    if number_of_returns is None:
        number_of_returns = np.full(n, 2, dtype=np.uint8)
    las.number_of_returns = number_of_returns

    if intensities is None:
        intensities = (np.arange(n, dtype=np.uint16) + 1) * 10  # 10, 20, ..., 120
    las.intensity = intensities

    return las


@pytest.fixture
def build_synthetic_las() -> Callable[..., laspy.LasData]:
    """Factory fixture; tests can override defaults via kwargs."""
    return _build_synthetic_las


@pytest.fixture
def synthetic_las_path(tmp_path) -> Path:
    """Default 12-point synthetic LAS written to ``tmp_path/synth.las``."""
    las = _build_synthetic_las()
    out = tmp_path / "synth.las"
    las.write(str(out))
    return out
