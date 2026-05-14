"""LAScatalog port — tile-based wall-to-wall ITS processing.

PORT NOTE
---------
Adapted from lidR's ``LAScatalog`` family (``R/LAScatalog-class.R``,
``R/catalog_apply.R``, ``R/catalog_options.R``,
``R/catalog_processing_engine.R``) and the per-tile buffer mechanics in
``R/LAScatalog-process.R`` (L42-180). The pylidar Phase 6 port covers
the workflow shape — tile enumeration, buffered loading, parallel
dispatch, per-tile output — rather than lidR's full feature surface
(no on-the-fly retiling, no progress bars, no auto-thread plan
detection).

* :class:`LAScatalog` enumerates tiles by file path. Per-tile bounds are
  read from the LAS header lazily and cached. ``buffer > 0`` means each
  tile gets neighbor points within ``buffer`` metres of its bbox for
  context (used by CHM smoothing and watershed; matches lidR's
  ``opt_chunk_buffer`` semantic at ``R/catalog_options.R:142-166``).
* :meth:`LAScatalog.map_tiles` is the workhorse. It builds one
  :class:`TileContext` per tile, then either runs them serially
  (``n_workers=1``) or via joblib's ``Parallel`` (``n_workers>1``).
  joblib is declared in the ``catalog`` optional-dependency bucket
  (per Q7 decision, 2026-05-11); the import is deferred so
  ``import pylidar.catalog`` doesn't fail when the extra is absent.
* :class:`TileContext` carries the tile path, index, bbox, buffer
  width, neighbor paths (spatially overlapping when buffered), and a
  helper :meth:`TileContext.load` that returns
  ``(las, unbuffered_mask)``. The mask identifies which points fall
  inside the **core** tile bbox vs. the buffer ring — workflows write
  only the core back to disk so seam regions aren't duplicated.
* Cross-tile tree-ID uniqueness is **handled at the writer layer**, not
  the catalog layer. Use
  :func:`pylidar.io.write_las_with_treeid` with
  ``uniqueness="bitmerge"`` (or ``"gpstime"``) per tile; the apex-derived
  float64 IDs are globally unique without any cross-worker coordination
  (lidR ``R/segment_trees.R:78-108`` shows the same separation —
  ``segment_trees`` writes per-LAS, ``LAScatalog`` only iterates).
  ``uniqueness="incremental"`` would yield collisions across tiles and
  is rejected by the convenience helper
  :func:`segment_trees_catalog`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

import laspy
import numpy as np
from numpy.typing import NDArray

from . import io as _io

__all__ = [
    "LAScatalog",
    "TileContext",
    "segment_trees_catalog",
]

_T = TypeVar("_T")
Bounds = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------- LAScatalog

class LAScatalog:
    """A collection of LAS / LAZ tiles for wall-to-wall ITS processing.

    Parameters
    ----------
    files : sequence of path-like
        Input tile paths. Each must point to an existing LAS or LAZ file.
    buffer : float, default 0.0
        Per-tile buffer width in world units (metres). When ``> 0``,
        :meth:`TileContext.load` pulls in neighbor points within this
        distance of the tile's bbox so border algorithms (CHM smoothing,
        watershed plateaus) see continuous context.

    Attributes
    ----------
    files : tuple[Path, ...]
    buffer : float
    tile_bounds : tuple[Bounds, ...]
        Lazy; reads each file's header on first access.
    """

    def __init__(
        self,
        files: Sequence[Union[str, Path]],
        *,
        buffer: float = 0.0,
    ) -> None:
        if not files:
            raise ValueError("LAScatalog requires at least one input file")
        if float(buffer) < 0:
            raise ValueError("buffer must be non-negative")
        resolved: list[Path] = []
        for f in files:
            p = Path(f)
            if not p.exists():
                raise FileNotFoundError(f"LAScatalog: file not found: {p}")
            resolved.append(p)
        self.files: tuple[Path, ...] = tuple(resolved)
        self.buffer: float = float(buffer)
        self._bounds_cache: Optional[tuple[Bounds, ...]] = None

    @property
    def n_tiles(self) -> int:
        return len(self.files)

    def __len__(self) -> int:
        return self.n_tiles

    def __iter__(self) -> Iterator[Path]:
        return iter(self.files)

    @property
    def tile_bounds(self) -> tuple[Bounds, ...]:
        """Per-tile ``(xmin, ymin, xmax, ymax)`` from the LAS headers."""
        if self._bounds_cache is None:
            self._bounds_cache = tuple(_read_bounds(p) for p in self.files)
        return self._bounds_cache

    def neighbors_of(self, tile_index: int) -> tuple[Path, ...]:
        """Tiles whose bbox overlaps tile ``i``'s buffered bbox.

        The self-tile is excluded. An empty tuple means either ``buffer == 0``
        or no other tile lies within ``buffer`` of this one.
        """
        if not 0 <= tile_index < self.n_tiles:
            raise IndexError(f"tile_index {tile_index} out of range")
        if self.buffer <= 0:
            return ()
        b = self.tile_bounds[tile_index]
        out: list[Path] = []
        for j, bj in enumerate(self.tile_bounds):
            if j == tile_index:
                continue
            if _bounds_overlap_with_buffer(b, bj, self.buffer):
                out.append(self.files[j])
        return tuple(out)

    def make_context(self, tile_index: int) -> "TileContext":
        if not 0 <= tile_index < self.n_tiles:
            raise IndexError(f"tile_index {tile_index} out of range")
        return TileContext(
            tile_path=self.files[tile_index],
            tile_index=tile_index,
            tile_bounds=self.tile_bounds[tile_index],
            buffer=self.buffer,
            neighbor_paths=self.neighbors_of(tile_index),
        )

    def map_tiles(
        self,
        fn: Callable[["TileContext"], _T],
        *,
        n_workers: int = 1,
    ) -> list[_T]:
        """Apply ``fn(context)`` to every tile.

        ``n_workers == 1`` runs serially in this process; ``n_workers > 1``
        dispatches via :mod:`joblib` (raises :class:`ImportError` if joblib
        isn't installed — install via ``pip install pylidar[catalog]``).
        """
        contexts = [self.make_context(i) for i in range(self.n_tiles)]
        if int(n_workers) <= 1:
            return [fn(ctx) for ctx in contexts]
        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pylidar.catalog parallel execution requires joblib. "
                "Install via `pip install pylidar[catalog]` "
                "(or pin n_workers=1 to stay in-process)."
            ) from exc
        return list(
            joblib.Parallel(n_jobs=int(n_workers))(
                joblib.delayed(fn)(ctx) for ctx in contexts
            )
        )


# ---------------------------------------------------------------- TileContext

@dataclass(frozen=True)
class TileContext:
    """Per-tile execution context handed to :meth:`LAScatalog.map_tiles` callbacks."""

    tile_path: Path
    tile_index: int
    tile_bounds: Bounds
    buffer: float
    neighbor_paths: tuple[Path, ...] = ()

    @property
    def buffered_bounds(self) -> Bounds:
        xmin, ymin, xmax, ymax = self.tile_bounds
        return (xmin - self.buffer, ymin - self.buffer,
                xmax + self.buffer, ymax + self.buffer)

    def load(self, **read_kwargs) -> tuple[laspy.LasData, NDArray[np.bool_]]:
        """Load this tile + buffer ring; return ``(las, core_mask)``.

        ``core_mask`` is a length-N bool array marking points inside the
        **unbuffered** tile bbox. Workflows should write only the core
        back to disk (or pass it through to the writer's ``tree_id``) so
        each input point appears in exactly one tile's output.

        ``read_kwargs`` are forwarded to :func:`pylidar.io.read_las`
        (filter kwargs only — ``select`` and ``laslib_filter`` are
        handled at the catalog level, not here).

        Implementation note
        -------------------
        v0 simplification: when neighbor points exist, we **rebuild** a
        :class:`laspy.LasData` from the concatenated *standard* fields
        (XYZ + return_number + number_of_returns + classification +
        intensity + optional gps_time and RGB). Extra-byte dimensions
        from neighbor tiles are dropped because LAS extra-byte
        descriptors may not be identical across tiles. Downstream
        per-tile output is then written from the **original** tile's
        :class:`laspy.LasData` (not this buffered one) — see
        :func:`segment_trees_catalog` for how the core_mask is applied.
        """
        las = _io.read_las(str(self.tile_path), **read_kwargs)
        if not self.neighbor_paths or self.buffer <= 0:
            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)
            mask = _core_mask(x, y, self.tile_bounds)
            return las, mask

        bx_min, by_min, bx_max, by_max = self.buffered_bounds
        merged_las = _clone_las(las)
        for n_path in self.neighbor_paths:
            n_las = _io.read_las(
                str(n_path),
                keep_xy=(bx_min, by_min, bx_max, by_max),
                **read_kwargs,
            )
            if len(n_las.x) == 0:
                continue
            merged_las = _concat_las(merged_las, n_las)

        x = np.asarray(merged_las.x, dtype=np.float64)
        y = np.asarray(merged_las.y, dtype=np.float64)
        mask = _core_mask(x, y, self.tile_bounds)
        return merged_las, mask


# ---------------------------------------------------------------- segment_trees_catalog

def segment_trees_catalog(
    catalog: LAScatalog,
    *,
    locate_fn: Callable[[laspy.LasData], NDArray],
    segment_fn: Callable[[laspy.LasData, NDArray], NDArray],
    output_dir: Union[str, Path],
    uniqueness: str = "bitmerge",
    n_workers: int = 1,
    output_suffix: str = "_seg.las",
    read_kwargs: Optional[dict] = None,
) -> list[Path]:
    """Convenience wrapper: run an ITS pipeline per tile, write to disk.

    Workflow per tile:

    1. Load the tile + buffer ring via :meth:`TileContext.load`.
    2. ``locate_fn(las_buffered) -> treetops`` (whatever the user's
       locator emits; passed straight to ``segment_fn``).
    3. ``segment_fn(las_buffered, treetops) -> int32 tree_id`` (one
       label per point in ``las_buffered``).
    4. Drop buffer-ring points (``~core_mask``).
    5. Write the **core** points via
       :func:`pylidar.io.write_las_with_treeid` to
       ``output_dir/<stem><output_suffix>``.

    ``uniqueness`` must be ``"bitmerge"`` or ``"gpstime"`` — these are
    the only modes whose IDs are globally unique without cross-worker
    coordination (lidR ``segment_trees.R:78-108``). ``"incremental"``
    raises :class:`ValueError`.
    """
    if uniqueness not in {"bitmerge", "gpstime"}:
        raise ValueError(
            f"segment_trees_catalog: uniqueness={uniqueness!r} would collide "
            f"across tiles; use 'bitmerge' or 'gpstime' for catalog workflows."
        )
    # Output filenames are derived from tile-path stems. If two inputs share
    # a stem (e.g. dir_a/tile_001.las and dir_b/tile_001.las), the later
    # worker would silently overwrite the earlier output. Detect upfront.
    stems = [p.stem for p in catalog.files]
    seen: dict[str, Path] = {}
    duplicates: list[tuple[str, Path, Path]] = []
    for path, stem in zip(catalog.files, stems):
        if stem in seen:
            duplicates.append((stem, seen[stem], path))
        else:
            seen[stem] = path
    if duplicates:
        details = "; ".join(
            f"{s!r} in {a} and {b}" for s, a, b in duplicates
        )
        raise ValueError(
            "segment_trees_catalog: input tiles share file-stem(s); "
            "outputs would silently overwrite each other. "
            f"Conflicts: {details}"
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rk = dict(read_kwargs or {})

    def _run(ctx: TileContext) -> Path:
        las_buf, core_mask = ctx.load(**rk)
        treetops = locate_fn(las_buf)
        labels = segment_fn(las_buf, treetops)
        labels = np.asarray(labels)
        if labels.shape[0] != len(las_buf.x):
            raise ValueError(
                f"segment_fn returned {labels.shape[0]} labels for "
                f"{len(las_buf.x)} points in tile {ctx.tile_path.name}"
            )
        labels_int32 = labels.astype(np.int32, copy=False)

        # Apex-based uniqueness MUST run on the buffered cloud so each
        # tree's apex (max-z point) is found across all points it spans,
        # including any in neighbor tiles' buffer rings. Doing this on
        # the core-only slice would let a tree's apex differ between the
        # two tiles that share it, and the derived bitmerge / gpstime IDs
        # would not match (the very thing cross-tile uniqueness is for).
        # Mirrors lidR R/segment_trees.R:51-108 which closes over the
        # current LAS (which lidR's catalog engine has already
        # constructed with the buffer ring).
        float_ids = _io._apply_uniqueness(
            labels_int32, las_buf, uniqueness=uniqueness
        )

        # Re-load the original (unbuffered) LAS so VLRs/CRS/header are
        # preserved exactly. Buffered load placed original-tile points
        # first in `_concat_las`, so float_ids[:n_original] is the
        # original tile's labels in the original order.
        original = _io.read_las(str(ctx.tile_path), **rk)
        n_original = len(original.x)
        core_in_original = core_mask[:n_original]
        na = float(np.finfo(np.float64).tiny)
        out_ids = np.where(
            core_in_original,
            float_ids[:n_original],
            na,
        ).astype(np.float64)

        out_path = out_dir / f"{ctx.tile_path.stem}{output_suffix}"
        # Writer sees float64 → passthrough (no second apex recomputation).
        _io.write_las_with_treeid(
            original,
            out_ids,
            out_path,
            uniqueness=uniqueness,
        )
        return out_path

    return catalog.map_tiles(_run, n_workers=n_workers)


# ---------------------------------------------------------------- internals

def _read_bounds(path: Path) -> Bounds:
    """Read (xmin, ymin, xmax, ymax) from a LAS header without loading points."""
    with laspy.open(str(path)) as fh:
        h = fh.header
        return (float(h.x_min), float(h.y_min), float(h.x_max), float(h.y_max))


def _bounds_overlap_with_buffer(a: Bounds, b: Bounds, buffer: float) -> bool:
    """Do bbox ``a`` expanded by ``buffer`` and bbox ``b`` overlap?"""
    a_xmin, a_ymin, a_xmax, a_ymax = a
    b_xmin, b_ymin, b_xmax, b_ymax = b
    return not (
        b_xmax < a_xmin - buffer
        or b_xmin > a_xmax + buffer
        or b_ymax < a_ymin - buffer
        or b_ymin > a_ymax + buffer
    )


def _core_mask(x: NDArray, y: NDArray, bounds: Bounds) -> NDArray[np.bool_]:
    xmin, ymin, xmax, ymax = bounds
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def _clone_las(las: laspy.LasData) -> laspy.LasData:
    """Return a deep clone of ``las`` via a BytesIO round-trip.

    Same pattern as :func:`pylidar.io.write_las_with_treeid`: laspy 2.x
    LasData can't be ``copy.deepcopy``'d, so we serialize/deserialize.
    """
    import io as _stdio
    buf = _stdio.BytesIO()
    las.write(buf)
    buf.seek(0)
    return laspy.read(buf)


# Fields we attempt to carry across the buffer concat. We probe each via
# attribute access (raises AttributeError on absent dims) rather than
# checking `point_format.dimension_names` — that table stores 'X/Y/Z'
# uppercase and uses ``scan_angle_rank`` for PF<6 / ``scan_angle`` for
# PF6+, neither of which matches the lowercase scaled accessors that
# the rest of the code uses. Gating on dimension_names silently skipped
# XYZ here, which would have left buffer points at ``(0, 0, 0)`` — a
# pre-fix bug that broke cross-tile apex uniqueness in turn.
# Extra-byte dimensions are deliberately not propagated (they may have
# inconsistent descriptors across a heterogeneous catalog).
_CONCAT_FIELDS = (
    "x", "y", "z",
    "return_number", "number_of_returns",
    "classification", "intensity",
    "gps_time", "red", "green", "blue", "nir",
    "scan_angle", "scan_angle_rank", "scanner_channel",
    "user_data", "point_source_id",
    "edge_of_flight_line", "scan_direction_flag",
)


def _concat_las(base: laspy.LasData, addition: laspy.LasData) -> laspy.LasData:
    """Concatenate ``addition``'s points onto ``base``.

    Both must share the same point format. We rebuild the underlying
    point record so laspy emits a single consistent buffer. Extra-byte
    dimensions on ``base`` are preserved (read from the original tile);
    extras on ``addition`` are dropped.
    """
    if base.point_format.id != addition.point_format.id:
        raise ValueError(
            f"catalog buffer load: neighbor point_format "
            f"{addition.point_format.id} != tile point_format "
            f"{base.point_format.id}"
        )

    n_base = len(base.x)
    n_add = len(addition.x)
    n_total = n_base + n_add
    if n_add == 0:
        return base

    # laspy 2.x exposes `points.resize(n)`, which zero-extends in-place.
    # We clone the base (so the original LasData isn't mutated), resize
    # to n_total, then overwrite each dim with the concatenated values.
    out = _clone_las(base)
    out.points.resize(n_total)

    for name in _CONCAT_FIELDS:
        try:
            base_arr = np.asarray(getattr(base, name))
        except AttributeError:
            continue  # dim isn't part of `base`'s point format
        try:
            add_arr = np.asarray(getattr(addition, name))
        except AttributeError:
            add_arr = np.zeros(n_add, dtype=base_arr.dtype)
        # XYZ goes through laspy's scale-aware accessors; world-coordinate
        # values round-trip without re-encoding back to int32 manually.
        setattr(out, name, np.concatenate([base_arr, add_arr]))

    return out
