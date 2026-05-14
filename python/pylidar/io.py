"""LAS / LAZ IO with lidR-shaped `select` DSL, `point_mask` filtering, and
``write_las_with_treeid``.

PORT NOTE
---------
Adapted from lidR ``R/io_readLAS.R`` (L5-260), ``R/filters.R`` (L49-200),
and ``R/segment_trees.R::segment_trees.LAS`` L37-108 (the
``add_lasattribute_manual`` finishing step that persists ``treeID`` as a
LAS extra-byte dimension). Translation choices, with the lidR semantics
they're matched against:

* ``select`` accepts the lidR letter DSL (``"xyziar"``, ``"*"``, ``"* -i -a"``,
  ``"xyz0"``) and a Python list (``["X", "Y", "Z"]``). The character → laspy
  field map is the one documented at ``io_readLAS.R:16-26``. ``X/Y/Z`` are
  always loaded. Missing requested fields are silent-skipped (mirrors lidR).
  ``W`` (full waveform) is **not** supported in this phase; passing it raises.
* In v0 the ``select`` DSL **does not actually drop standard point dimensions
  nor extra-byte columns** from the returned :class:`laspy.LasData`. laspy's
  point record is a packed structure where dropping a single column would
  require rebuilding the point format and re-encoding every record; that's
  invasive and not justified at this scope. Critically we also do **not**
  prune the ExtraBytes VLR descriptors, even though lidR
  (``io_readLAS.R:264-268``) does — pruning the VLR alone would create
  a metadata/data inconsistency (the point record still carries the bytes
  but the VLR claims it doesn't), which laspy serializes back as anonymous
  extras with the original names lost. The conservative v0 contract is
  "select records intent, doesn't mutate"; the parsed selection is exposed
  via ``las.user_select`` so downstream consumers can check intent. A
  follow-up phase can add coordinated drop-both-or-neither once a real
  user need lands.
* ``filter_kwargs`` (``keep_first=True``, ``drop_class=[7]``,
  ``drop_z_below=0.0`` …) are applied as **post-load numpy boolean masks**.
  This deliberately diverges from lidR/rlas, which streams raw LASlib
  ``-keep_*`` / ``-drop_*`` filter strings through the C++ rlas layer at
  read time. The Python ecosystem doesn't have a binding to LASlib's filter
  parser, and ``readLAS`` itself doesn't chunk (chunking lives at the
  ``LAScatalog`` layer — see Phase 6); load-then-mask is functionally
  equivalent for RAM budgets ≤ a few GB. Multiple kwargs AND-combine,
  matching ``filter_poi``'s NSE semantics (``filters.R:49-78``); NaN
  predicate results are coerced to ``False`` (``filters.R:73``).
* ``laslib_filter="..."`` is the power-user escape hatch for a raw LASlib
  string. Phase 2 only stubs it: a non-empty value raises
  :class:`NotImplementedError` pointing the user at the kwarg vocabulary.
* The kwarg vocabulary is the lidR named-macro list
  (``filter_first``, ``filter_ground``, ``remove_noise`` …) flattened into
  ``keep_*`` / ``drop_*`` parameters — we do **not** introduce a
  ``pylidar.filters`` module or replicate the R NSE machinery
  (``filter_poi``); ``point_mask(las, **kwargs)`` is the unified entry.
* :func:`write_las_with_treeid` clones ``las_in`` via a BytesIO round-trip
  (laspy's records can't be ``copy.deepcopy``'d — its ``__getattr__``
  recurses), then ADDs a ``treeID`` extra-byte dimension and writes.
  The extra-byte ``type`` is driven by ``tree_id.dtype`` AND
  ``uniqueness``:

  * ``tree_id.dtype == int32`` + ``uniqueness="incremental"`` → persist
    as LAS spec type 6 (int32), NA sentinel ``np.iinfo(np.int32).max``.
  * ``tree_id.dtype == int32`` + ``uniqueness ∈ {"gpstime", "bitmerge"}``
    → recompute per-tree apex IDs from ``las_in`` (max-z point per tree,
    1:1 with lidR ``segment_trees.R:51-63`` ``tapex``/``xyapex``
    closures), persist as LAS spec type 10 (float64), NA sentinel
    ``np.finfo(np.float64).tiny`` (R's ``.Machine$double.xmin``).
  * ``tree_id.dtype == float64`` → "already finalized" path; persist
    as-is regardless of uniqueness param (Phase 4's
    :class:`Treetops.tree_id` propagation paths use this).

  Apex tie-breakers match lidR ``segment_trees.R:51-63``: gpstime ties
  → lowest gpstime; bitmerge ties → lowest x. Points whose
  ``tree_id == 0`` (pylidar's no-tree sentinel) or whose tree_id equals
  the input-dtype NA sentinel are skipped from grouping and written
  with the output-dtype NA sentinel — out-of-grid points and in-grid
  no-tree points collapse to the LAS-spec NA on disk.
  RGB is preserved when present in the input point format; if the user
  passes ``rgb=`` AND the input point format lacks RGB, we raise —
  upgrading the point format would force re-encoding every point
  record and is out of scope.
"""

from __future__ import annotations

import io as _io
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import laspy
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "read_las",
    "point_mask",
    "write_las_with_treeid",
    "FilterKwargError",
]
# ``parse_select`` is exposed at module level for backward compatibility
# with tests but is **not** in ``__all__`` — it's a parser helper, not
# part of the stable public surface. Phase 5 audit fix #4: prior versions
# returned a 3-tuple; the current 4-tuple shape is internal.


# Char → canonical laspy field name. Sourced from lidR ``io_readLAS.R:16-26``.
# Where a single LAS attribute has aliases across point formats (e.g. legacy
# ``ScanAngleRank`` int8 vs PDRF6+ ``ScanAngle`` int16), we map to the laspy
# accessor that papers over the difference: ``las.scan_angle`` exists from
# laspy 2.4+ and dispatches to whichever underlying dim the format provides.
_CHAR_TO_FIELD: dict[str, str] = {
    # XYZ — always loaded
    "x": "X",
    "y": "Y",
    "z": "Z",
    # standard
    "t": "gps_time",
    "a": "scan_angle",
    "i": "intensity",
    "n": "number_of_returns",
    "r": "return_number",
    "c": "classification",
    "s": "synthetic",
    "k": "key_point",
    "w": "withheld",
    "o": "overlap",  # PDRF 6+
    "u": "user_data",
    "p": "point_source_id",
    "e": "edge_of_flight_line",
    "d": "scan_direction_flag",
    # uppercase
    "R": "red",
    "G": "green",
    "B": "blue",
    "N": "nir",
    "C": "scanner_channel",  # PDRF 6+
}

# Sentinel: 'W' is full-waveform; pylidar Phase 2 does not support it.
_FWF_CHAR = "W"

# Selection always includes X/Y/Z regardless of user input.
_MANDATORY = frozenset({"X", "Y", "Z"})


class FilterKwargError(TypeError):
    """Raised when ``read_las`` / ``point_mask`` receives an unknown kwarg."""


def parse_select(
    select: Union[str, Sequence[str], None],
) -> tuple[frozenset[str], frozenset[int], bool, frozenset[int]]:
    """Parse the lidR-style ``select`` DSL into laspy field requests.

    Returns
    -------
    fields : frozenset[str]
        Requested laspy field names (always includes X, Y, Z).
    extras_kept : frozenset[int]
        Explicitly-requested 1-based extra-byte ordinals (only meaningful
        when ``all_extras=False``). Per lidR convention ``1`` → first extra
        byte.
    all_extras : bool
        True when the input contained ``*`` or ``0`` (or was ``None``,
        which defaults to ``"*"``). The caller should preserve every extra
        byte EXCEPT those listed in ``extras_dropped``.
    extras_dropped : frozenset[int]
        Extra-byte ordinals explicitly negated (e.g. ``"* -1"`` →
        ``extras_dropped={1}``). Only meaningful when ``all_extras=True``.
    """
    if select is None:
        return (
            frozenset(_MANDATORY | _all_standard_field_names()),
            frozenset(),
            True,
            frozenset(),
        )

    if isinstance(select, str):
        return _parse_select_string(select)

    return _parse_select_sequence(select)


def _all_standard_field_names() -> set[str]:
    return {field for field in _CHAR_TO_FIELD.values()}


def _parse_select_string(
    s: str,
) -> tuple[frozenset[str], frozenset[int], bool, frozenset[int]]:
    fields: set[str] = set(_MANDATORY)
    extras_kept: set[int] = set()
    extras_dropped: set[int] = set()
    all_extra = False

    # Tokenize: whitespace separates `*`, `-x`, char-runs.
    tokens = s.split()
    for tok in tokens:
        if tok == "*":
            fields.update(_all_standard_field_names())
            all_extra = True
            continue
        if tok.startswith("-") and len(tok) > 1:
            for ch in tok[1:]:
                _drop_char(
                    ch, fields, extras_kept, extras_dropped, all_extra=all_extra
                )
                if ch == "0":
                    # `-0` clears everything: switch from all-extras to no-extras
                    all_extra = False
                    extras_kept.clear()
                    extras_dropped.clear()
            continue
        for ch in tok:
            _add_char(ch, fields, extras_kept)
            if ch == "0":
                all_extra = True

    return frozenset(fields), frozenset(extras_kept), all_extra, frozenset(extras_dropped)


def _add_char(ch: str, fields: set[str], extras: set[int]) -> None:
    if ch == _FWF_CHAR:
        raise NotImplementedError(
            "select: full-waveform ('W') is not supported in this phase"
        )
    if ch == "0":
        return
    if ch.isdigit():
        extras.add(int(ch))
        return
    if ch not in _CHAR_TO_FIELD:
        raise ValueError(f"select: unknown character {ch!r}")
    fields.add(_CHAR_TO_FIELD[ch])


def _drop_char(
    ch: str,
    fields: set[str],
    extras_kept: set[int],
    extras_dropped: set[int],
    *,
    all_extra: bool,
) -> None:
    if ch == _FWF_CHAR:
        return
    if ch == "0":
        # Caller resets all_extra / extras_kept / extras_dropped after this.
        return
    if ch.isdigit():
        ordinal = int(ch)
        if all_extra:
            # Within an "all extras" context, `-N` records a drop request.
            extras_dropped.add(ordinal)
        else:
            extras_kept.discard(ordinal)
        return
    if ch not in _CHAR_TO_FIELD:
        raise ValueError(f"select: unknown character {ch!r}")
    field = _CHAR_TO_FIELD[ch]
    if field in _MANDATORY:
        # X/Y/Z are mandatory per lidR; silently keep them.
        return
    fields.discard(field)


def _parse_select_sequence(
    seq: Sequence[str],
) -> tuple[frozenset[str], frozenset[int], bool, frozenset[int]]:
    fields: set[str] = set(_MANDATORY)
    extras_kept: set[int] = set()
    all_extra = False
    for name in seq:
        if not isinstance(name, str):
            raise TypeError(f"select: sequence entries must be str, got {type(name).__name__}")
        if name == "*":
            fields.update(_all_standard_field_names())
            all_extra = True
            continue
        if name.lower() == "all_extra_bytes":
            all_extra = True
            continue
        # Allow either canonical laspy name or single-char DSL.
        if len(name) == 1 and (name in _CHAR_TO_FIELD or name == _FWF_CHAR or name.isdigit() or name == "0"):
            _add_char(name, fields, extras_kept)
            if name == "0":
                all_extra = True
            continue
        # Treat as canonical laspy field name (case-insensitive match against table values).
        target = name if name in _CHAR_TO_FIELD.values() else _normalize_field_name(name)
        if target is None:
            raise ValueError(f"select: unknown field name {name!r}")
        fields.add(target)
    return frozenset(fields), frozenset(extras_kept), all_extra, frozenset()


def _normalize_field_name(name: str) -> Union[str, None]:
    canonical = name.lower()
    # Common spellings
    aliases = {
        "intensity": "intensity",
        "scan_angle": "scan_angle",
        "scan_angle_rank": "scan_angle",
        "return_number": "return_number",
        "number_of_returns": "number_of_returns",
        "classification": "classification",
        "gps_time": "gps_time",
        "user_data": "user_data",
        "point_source_id": "point_source_id",
        "edge_of_flight_line": "edge_of_flight_line",
        "scan_direction_flag": "scan_direction_flag",
        "synthetic": "synthetic",
        "key_point": "key_point",
        "withheld": "withheld",
        "overlap": "overlap",
        "red": "red",
        "green": "green",
        "blue": "blue",
        "nir": "nir",
        "scanner_channel": "scanner_channel",
        "x": "X",
        "y": "Y",
        "z": "Z",
    }
    return aliases.get(canonical)


# ---------------------------------------------------------------- read_las

def read_las(
    path: Union[str, Path],
    *,
    select: Union[str, Sequence[str], None] = None,
    laslib_filter: Union[str, None] = None,
    **filter_kwargs,
) -> laspy.LasData:
    """Read a LAS / LAZ file with lidR-shaped select + filter semantics.

    Parameters
    ----------
    path : str | Path
        LAS or LAZ file. LAZ requires ``lazrs`` (declared in dev extras).
    select : str | sequence[str] | None
        lidR letter DSL (``"xyziar"``, ``"*"``, ``"* -i -a"``, ``"xyz0"``) or
        a sequence of canonical laspy field names. ``None`` means
        ``"*"`` (load everything). XYZ are always retained.
    laslib_filter : str | None
        Power-user escape for a raw LASlib filter string. Phase 2 raises
        :class:`NotImplementedError`; use the kwargs below instead.
    **filter_kwargs
        Boolean mask predicates AND-combined; see :func:`point_mask`.

    Returns
    -------
    laspy.LasData
        With the ``filter_kwargs`` mask applied to ``las.points``. The parsed
        selection is exposed via ``las.user_select`` (a ``frozenset[str]``)
        for downstream consumers; standard point dimensions are not actually
        dropped in v0 (see PORT NOTE).
    """
    if laslib_filter is not None and laslib_filter != "":
        raise NotImplementedError(
            "laslib_filter is not implemented in this phase. "
            "Use kwargs (keep_first=True, drop_class=[7], drop_z_below=0.0, ...) instead."
        )

    # Parse for validation + intent recording. We deliberately do NOT mutate
    # the LasData (neither standard dims nor ExtraBytes VLR) — see PORT NOTE
    # for the metadata/data inconsistency that VLR-only pruning would cause.
    requested_fields, _extras_kept, _all_extras, _extras_dropped = parse_select(select)

    las = laspy.read(str(path))

    if filter_kwargs:
        mask = point_mask(las, **filter_kwargs)
        las.points = las.points[mask]

    las.user_select = requested_fields
    return las


# ---------------------------------------------------------------- point_mask

# Recognized kwargs and the column they consult (for error messages).
_KWARG_HELP = {
    "keep_first": "ReturnNumber",
    "keep_last": "ReturnNumber, NumberOfReturns",
    "keep_single": "NumberOfReturns",
    "keep_firstofmany": "ReturnNumber, NumberOfReturns",
    "keep_firstlast": "ReturnNumber, NumberOfReturns",
    "keep_ground": "Classification",
    "keep_class": "Classification",
    "drop_class": "Classification",
    "drop_z_below": "Z",
    "drop_z_above": "Z",
    "drop_intensity_below": "Intensity",
    "drop_intensity_above": "Intensity",
    "keep_xy": "X, Y",
}


def point_mask(las: laspy.LasData, /, **kwargs) -> NDArray[np.bool_]:
    """Build a boolean keep-mask from lidR-named-macro predicates.

    Multiple kwargs AND-combine. NaN predicate results coerce to ``False``
    (matches ``filters.R::lasfilter_`` line 73). Unknown kwargs raise
    :class:`FilterKwargError`.

    Recognized kwargs (lidR ``filter_*`` named-macro vocabulary, flattened):

    ============================  ==============================================
    keep_first=True               ReturnNumber == 1                              (filter_first)
    keep_last=True                ReturnNumber == NumberOfReturns                (filter_last)
    keep_single=True              NumberOfReturns == 1                           (filter_single)
    keep_firstofmany=True         (NumberOfReturns > 1) & (ReturnNumber == 1)    (filter_firstofmany)
    keep_firstlast=True           filter_first | filter_last                     (filter_firstlast)
    keep_ground=True              Classification == 2                            (filter_ground)
    keep_class=int | sequence     Classification ∈ values                        (-keep_class N…)
    drop_class=int | sequence     Classification ∉ values                        (remove_noise / -drop_class)
    drop_z_below=float            las.z >= value                                 (-drop_z_below N)
    drop_z_above=float            las.z <= value                                 (-drop_z_above N)
    drop_intensity_below=int      las.intensity >= value                         (-drop_intensity_below N)
    drop_intensity_above=int      las.intensity <= value                         (-drop_intensity_above N)
    keep_xy=(xmin, ymin, xmax, ymax)
                                  bounding-box (xmax inclusive)                  (-keep_xy)
    ============================  ==============================================
    """
    unknown = set(kwargs) - set(_KWARG_HELP)
    if unknown:
        raise FilterKwargError(
            f"point_mask: unknown kwargs {sorted(unknown)}. "
            f"Known: {sorted(_KWARG_HELP)}"
        )

    n = len(las.x)
    mask = np.ones(n, dtype=np.bool_)

    if kwargs.get("keep_first"):
        mask &= _coerce(np.asarray(las.return_number) == 1)
    if kwargs.get("keep_last"):
        mask &= _coerce(np.asarray(las.return_number) == np.asarray(las.number_of_returns))
    if kwargs.get("keep_single"):
        mask &= _coerce(np.asarray(las.number_of_returns) == 1)
    if kwargs.get("keep_firstofmany"):
        rn = np.asarray(las.return_number)
        nr = np.asarray(las.number_of_returns)
        mask &= _coerce((nr > 1) & (rn == 1))
    if kwargs.get("keep_firstlast"):
        rn = np.asarray(las.return_number)
        nr = np.asarray(las.number_of_returns)
        mask &= _coerce((rn == 1) | (rn == nr))
    if kwargs.get("keep_ground"):
        mask &= _coerce(np.asarray(las.classification) == 2)
    if "keep_class" in kwargs:
        values = _as_int_seq(kwargs["keep_class"], "keep_class")
        mask &= _coerce(np.isin(np.asarray(las.classification), values))
    if "drop_class" in kwargs:
        values = _as_int_seq(kwargs["drop_class"], "drop_class")
        mask &= _coerce(~np.isin(np.asarray(las.classification), values))
    if "drop_z_below" in kwargs:
        mask &= _coerce(np.asarray(las.z) >= float(kwargs["drop_z_below"]))
    if "drop_z_above" in kwargs:
        mask &= _coerce(np.asarray(las.z) <= float(kwargs["drop_z_above"]))
    if "drop_intensity_below" in kwargs:
        mask &= _coerce(np.asarray(las.intensity) >= int(kwargs["drop_intensity_below"]))
    if "drop_intensity_above" in kwargs:
        mask &= _coerce(np.asarray(las.intensity) <= int(kwargs["drop_intensity_above"]))
    if "keep_xy" in kwargs:
        bbox = kwargs["keep_xy"]
        if len(bbox) != 4:
            raise ValueError("keep_xy must be (xmin, ymin, xmax, ymax)")
        xmin, ymin, xmax, ymax = (float(v) for v in bbox)
        x = np.asarray(las.x)
        y = np.asarray(las.y)
        mask &= _coerce((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))

    return mask


def _coerce(arr: NDArray) -> NDArray[np.bool_]:
    """Coerce to bool, treating NaN as False (filters.R:73 contract)."""
    if arr.dtype == np.bool_:
        return arr
    out = np.asarray(arr, dtype=np.bool_)
    # If the source had NaN, np.bool_ cast would have raised; this branch is
    # for object/numeric arrays. We mirror lidR's `bools[is.na(bools)] <- FALSE`
    # by zero-filling; numpy's dtype-bool cast already maps NaN → False on the
    # rare path where a comparison produced object dtype.
    return out


def _as_int_seq(value: Union[int, Iterable[int]], name: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        return np.asarray([int(value)], dtype=np.int32)
    try:
        arr = np.asarray(list(value), dtype=np.int32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name}: must be int or sequence of int, got {value!r}") from exc
    return arr


# ---------------------------------------------------------------- write_las_with_treeid

_RGB_POINT_FORMATS = frozenset({2, 3, 5, 7, 8, 10})
_UNIQUENESS_MODES = frozenset({"incremental", "gpstime", "bitmerge"})


def write_las_with_treeid(
    las_in: laspy.LasData,
    tree_id: NDArray,
    out_path: Union[str, Path],
    *,
    rgb: Optional[NDArray] = None,
    uniqueness: str = "incremental",
) -> None:
    """Write ``las_in`` to ``out_path`` with a persistent ``treeID`` extra-dim.

    Preserves ``las_in``'s header (version, point_format, scales, offsets),
    every VLR (including the GeoKey / CRS VLRs), and every existing point
    dimension. Adds a single ``treeID`` extra-byte dimension whose
    LAS-spec type is driven by ``tree_id.dtype``:

    ============  =================  =================================
    tree_id.dtype LAS extra type     NA sentinel (no_data)
    ============  =================  =================================
    int32         ``"int"`` (6)      ``np.iinfo(np.int32).max``
    float64       ``"double"`` (10)  ``np.finfo(np.float64).tiny``
    ============  =================  =================================

    Mirrors lidR ``segment_trees.R:43-46/78/106``: int path uses
    ``.Machine$integer.max``; double path uses ``.Machine$double.xmin``.

    Parameters
    ----------
    las_in : laspy.LasData
        Source point cloud. All dimensions and VLRs are carried over.
    tree_id : (N,) int32 | float64
        Per-point label, length must equal ``len(las_in.x)``. Points
        outside any tree should be labelled with the NA sentinel
        (``np.iinfo(np.int32).max`` for int / ``np.finfo(np.float64).tiny``
        for double) or with ``0`` for the conventional "unsegmented"
        int sentinel.
    out_path : str | Path
        Destination LAS / LAZ file.
    rgb : (N, 3) uint16 | None
        Optional RGB triplet. The input point format must already support
        RGB (formats 2, 3, 5, 7, 8, 10); we don't upgrade the point format
        in v0. When ``rgb`` is ``None`` the input's existing RGB (if any)
        is preserved as-is.
    uniqueness : {'incremental', 'gpstime', 'bitmerge'}, default 'incremental'
        Records the lidR uniqueness mode in the extra-byte description
        string. Does not itself change the on-disk dtype; pick
        ``tree_id.dtype`` to drive that.
    """
    if uniqueness not in _UNIQUENESS_MODES:
        raise ValueError(
            f"uniqueness must be one of {sorted(_UNIQUENESS_MODES)}, got {uniqueness!r}"
        )

    if not isinstance(tree_id, np.ndarray):
        raise TypeError("tree_id must be a numpy ndarray")
    if tree_id.ndim != 1:
        raise ValueError("tree_id must be 1-D")
    n = len(las_in.x)
    if tree_id.shape[0] != n:
        raise ValueError(
            f"tree_id length {tree_id.shape[0]} != input point count {n}"
        )

    # Pass the numpy dtype directly so laspy writes LAS spec table 24 types
    # 6 (int = int32, 4 bytes) or 10 (double = float64, 8 bytes). Passing the
    # string ``"int"`` would route through laspy's 8-byte int path, which
    # is a wider type than lidR's ``add_lasattribute_manual(..., type="int")``
    # writes (rlas type 6 = 4-byte signed int).
    if tree_id.dtype == np.int32:
        if uniqueness == "incremental":
            eb_type = np.int32
            eb_no_data = int(np.iinfo(np.int32).max)
            tree_id_out = tree_id
        else:
            # Apex-based recomputation; output is float64 per
            # lidR segment_trees.R:78,106 (NA_value = .Machine$double.xmin).
            eb_type = np.float64
            eb_no_data = float(np.finfo(np.float64).tiny)
            tree_id_out = _apply_uniqueness(
                tree_id, las_in, uniqueness=uniqueness
            )
    elif tree_id.dtype == np.float64:
        # "Already finalized" path — Phase 4 propagation already encoded the
        # mode via Treetops.tree_id; persist as-is, no apex recomputation.
        eb_type = np.float64
        eb_no_data = float(np.finfo(np.float64).tiny)
        tree_id_out = tree_id
    else:
        raise TypeError(
            f"tree_id dtype must be int32 or float64, got {tree_id.dtype}"
        )

    pf_id = int(las_in.point_format.id)
    if rgb is not None:
        if pf_id not in _RGB_POINT_FORMATS:
            raise ValueError(
                f"rgb supplied but input point_format {pf_id} has no RGB channels; "
                f"supported formats: {sorted(_RGB_POINT_FORMATS)}"
            )
        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim != 2 or rgb_arr.shape != (n, 3):
            raise ValueError(
                f"rgb must have shape ({n}, 3), got {rgb_arr.shape}"
            )
        if rgb_arr.dtype != np.uint16:
            rgb_arr = rgb_arr.astype(np.uint16)

    # laspy 2.x can't copy.deepcopy a LasData (its __getattr__ recurses on
    # missing attributes); the BytesIO round-trip is the supported clone
    # path and preserves every VLR, header field, and point dimension.
    buf = _io.BytesIO()
    las_in.write(buf)
    buf.seek(0)
    las_out = laspy.read(buf)

    if "treeID" in las_out.point_format.dimension_names:
        raise ValueError(
            "input LAS already has a 'treeID' extra dimension; remove it before writing."
        )

    las_out.add_extra_dim(
        laspy.ExtraBytesParams(
            name="treeID",
            type=eb_type,
            description=f"tree id ({uniqueness})",
            no_data=[eb_no_data],
        )
    )
    las_out.treeID = tree_id_out.copy()

    if rgb is not None:
        las_out.red = rgb_arr[:, 0]
        las_out.green = rgb_arr[:, 1]
        las_out.blue = rgb_arr[:, 2]

    las_out.write(str(out_path))


# ---------------------------------------------------------------- uniqueness helpers
#
# 1:1 port of lidR ``R/segment_trees.R:51-108`` ``tapex`` / ``xyapex`` /
# bitmerge branch. Inputs are per-point int32 tree IDs (raw segmentation
# output), output is a per-point float64 array of apex-derived IDs.

_INT32_NA = np.iinfo(np.int32).max


def _apply_uniqueness(
    tree_id: NDArray[np.int32],
    las_in: laspy.LasData,
    *,
    uniqueness: str,
) -> NDArray[np.float64]:
    """Compute per-tree apex IDs (gpstime or bitmerge) and broadcast.

    Mirrors lidR ``segment_trees.R:68-108``: groups int32 ``tree_id``
    by value (skipping the no-tree sentinels 0 and ``INT32_NA``),
    finds the per-tree apex via ``max(z)`` with a deterministic
    tie-breaker (``gpstime``: lowest gps time; ``bitmerge``: lowest x),
    derives the per-tree ID from that apex, and returns a length-N
    float64 array.

    Points whose ``tree_id`` is a no-tree sentinel keep the LAS NA
    sentinel ``np.finfo(np.float64).tiny``.
    """
    z = np.asarray(las_in.z, dtype=np.float64)
    out = np.full(tree_id.shape[0], np.finfo(np.float64).tiny, dtype=np.float64)

    valid_mask = (tree_id != 0) & (tree_id != _INT32_NA)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return out

    ids_valid = tree_id[valid_idx]
    z_valid = z[valid_idx]

    if uniqueness == "gpstime":
        if not hasattr(las_in, "gps_time"):
            raise ValueError(
                "uniqueness='gpstime' requires the input LAS to carry gps_time "
                "(lidR segment_trees.R:17-18)."
            )
        gpstime = np.asarray(las_in.gps_time, dtype=np.float64)
        if not np.any(gpstime != 0.0):
            raise ValueError(
                "Impossible to compute unique IDs using gpstime: "
                "gpstime is not populated (lidR segment_trees.R:20-21)."
            )
        t_valid = gpstime[valid_idx]
        # lexsort: primary ids ASC, secondary z DESC, tertiary gpstime ASC.
        # First element per group → max-z with min-gpstime tie-breaker.
        order = np.lexsort((t_valid, -z_valid, ids_valid))
        ids_sorted = ids_valid[order]
        t_sorted = t_valid[order]
        unique_ids, first_idx = np.unique(ids_sorted, return_index=True)
        apex_t = t_sorted[first_idx]
        # Map each valid point's id to its apex_t.
        id_to_apex = dict(zip(unique_ids.tolist(), apex_t.tolist()))
        out[valid_idx] = np.fromiter(
            (id_to_apex[int(tid)] for tid in ids_valid),
            dtype=np.float64,
            count=ids_valid.size,
        )
        return out

    if uniqueness == "bitmerge":
        # lidR segment_trees.R:86-88: scale = header X/Y scale factor;
        # offset = header X/Y offset.
        scales = las_in.header.scales
        offsets = las_in.header.offsets
        xscale = float(scales[0])
        yscale = float(scales[1])
        if xscale <= 0.0 or yscale <= 0.0:
            raise ValueError(
                f"uniqueness='bitmerge' requires positive X/Y scales, "
                f"got X={xscale}, Y={yscale}"
            )
        xoffset = float(offsets[0])
        yoffset = float(offsets[1])
        x_full = np.asarray(las_in.x, dtype=np.float64)
        y_full = np.asarray(las_in.y, dtype=np.float64)
        x_valid = x_full[valid_idx]
        y_valid = y_full[valid_idx]
        # lexsort: primary ids ASC, secondary z DESC, tertiary x ASC.
        # First element per group → max-z apex with min-x tie-breaker.
        order = np.lexsort((x_valid, -z_valid, ids_valid))
        ids_sorted = ids_valid[order]
        x_sorted = x_valid[order]
        y_sorted = y_valid[order]
        unique_ids, first_idx = np.unique(ids_sorted, return_index=True)
        apex_x = x_sorted[first_idx]
        apex_y = y_sorted[first_idx]
        # Scale to int32 per lidR L94-95.
        xscaled = ((apex_x - xoffset) / xscale).astype(np.int32)
        yscaled = ((apex_y - yoffset) / yscale).astype(np.int32)
        from .locate_trees import bitmerge as _bitmerge
        apex_ids = _bitmerge(xscaled, yscaled)
        id_to_apex = dict(zip(unique_ids.tolist(), apex_ids.tolist()))
        out[valid_idx] = np.fromiter(
            (id_to_apex[int(tid)] for tid in ids_valid),
            dtype=np.float64,
            count=ids_valid.size,
        )
        return out

    # Defensive: caller already validated uniqueness ∈ _UNIQUENESS_MODES.
    raise ValueError(f"unsupported uniqueness mode: {uniqueness!r}")
