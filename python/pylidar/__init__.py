"""pylidar — individual tree segmentation algorithms ported from R lidR.

Public API (v0.1):
    set_log_callback(callable | None)   — wired through C++ core
    smooth_height(xyz, size, method, shape, sigma=None)   — Phase 1
    # Algorithm wrappers landing in later phases:
    #   locate_trees_lmf_chm(...) / locate_trees_lmf_points(...)
    #   segment_dalponte2016(...) / segment_silva2016(...)
    #   segment_li2012(...)
    #   segment_watershed(...)

Internal modules:
    pylidar._core       — nanobind extension; do not import directly
    pylidar._validate   — input dtype/shape/contiguity checks
    pylidar.segmentation — high-level wrappers, including the pure-Python
                           watershed implementation
"""

from __future__ import annotations

from . import _core
from ._core import set_log_callback
from .segmentation import smooth_height

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "set_log_callback",
    "smooth_height",
]
