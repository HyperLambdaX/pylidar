"""Shared test utilities.

Per spec §6, algorithm fixtures live as .npz files under tests/fixtures/.
Each carries ``inputs/*``, ``expected/*``, and ``meta/*`` keys; the
``load_fixture`` pytest fixture returns the union as a flat dict so callers
can unpack with explicit prefixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

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
