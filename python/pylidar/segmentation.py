"""High-level Python wrappers around :mod:`pylidar._core`.

Phase 0 placeholder: algorithm wrappers (``smooth_height``, ``locate_trees_*``,
``segment_*``) land in Phases 1–6. ``segment_watershed`` will be implemented
purely in Python on top of :mod:`skimage.segmentation` (spec §11; v0.1 keeps
watershed out of the C++ core).
"""

from __future__ import annotations

__all__: list[str] = []
