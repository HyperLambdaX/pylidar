"""One-shot fixture generator.

This script writes the .npz fixtures consumed by tests/. The arrays in the
``expected_*`` slot were hand-traced from the upstream lidR sources before
any implementation was run (spec §6 discipline); save-to-disk is a plain
serialization step, not a record-the-implementation step. Re-run with::

    uv run python tests/fixtures/_make_fixtures.py

after editing this file. The .npz outputs are committed alongside it; this
script is preserved purely as documentation of the hand-trace and so the
fixtures can be regenerated reproducibly.

For each fixture, the .npz contains:
    inputs/...    — the raw ndarrays handed to the algorithm under test.
    expected/...  — the hand-traced expected output(s).
    meta/source   — string referencing lidR source path + line range.
    meta/lidR_commit_ref — TBD until an R + lidR environment is wired up.
    meta/notes    — short narrative describing what the fixture covers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# --- dalponte2016 fixtures ---------------------------------------------------

def fx_dalponte2016_happy() -> dict:
    chm = np.array([
        [0, 0,  0, 0,  0],
        [0, 10, 6, 0,  0],
        [0, 6,  4, 6,  0],
        [0, 0,  6, 10, 0],
        [0, 0,  0, 0,  0],
    ], dtype=np.float64)
    seeds = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    return {
        "inputs/chm": chm,
        "inputs/seeds": seeds,
        "inputs/th_tree": np.float64(2.0),
        "inputs/th_seed": np.float64(0.45),
        "inputs/th_cr":   np.float64(0.55),
        "inputs/max_cr":  np.float64(10.0),
        "expected/regions": expected,
        "meta/source": "manual_derivation: lidR/src/C_dalponte2016.cpp:30-126",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Two seeds at (1,1)/id=1 and (3,3)/id=2; canopy 6/10 forms an "
            "X. Each seed grows by one ring along the 4-connected canopy "
            "where pz>th_cr*mhCrown holds; pass 2 has nothing to do."
        ),
    }


def fx_dalponte2016_degenerate() -> dict:
    """Single seed sitting in a too-shallow canopy: th_cr (0.55) gates every
    neighbour at z=5 because hSeed=10 ⇒ 5 ≯ 5.5. No growth. Pass exits in
    one iteration."""
    chm = np.array([
        [0, 0, 0,  0, 0],
        [0, 0, 5,  0, 0],
        [0, 5, 10, 5, 0],
        [0, 0, 5,  0, 0],
        [0, 0, 0,  0, 0],
    ], dtype=np.float64)
    seeds = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    expected = seeds.copy()
    return {
        "inputs/chm": chm,
        "inputs/seeds": seeds,
        "inputs/th_tree": np.float64(2.0),
        "inputs/th_seed": np.float64(0.45),
        "inputs/th_cr":   np.float64(0.55),
        "inputs/max_cr":  np.float64(10.0),
        "expected/regions": expected,
        "meta/source": "manual_derivation: lidR/src/C_dalponte2016.cpp:99-117",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Degenerate: seed(2,2)/id=1 with hSeed=10 surrounded by 5's. "
            "th_cr=0.55 ⇒ pz>5.5 required; pz=5 fails. No expansion. "
            "Output equals seeds."
        ),
    }


def fx_dalponte2016_corner_tie() -> dict:
    """Two seeds compete for the same neighbour pixel in one pass.

    Iteration order is row-major: seed1 at (1,1) iterates BEFORE seed2 at
    (1,3) within the same row. seed1 first claims (1,2) into Regiontemp;
    seed2 — reading the *unchanged* Region snapshot, which still has
    (1,2)=0 — overwrites Regiontemp(1,2)=2. seed2 wins.

    What this fixture validates: the row-major tie-break order on a shared
    neighbour. lidR has a related book-keeping drift (the loser's
    npixel/sum_height were incremented but never rolled back), but that
    state is internal to the algorithm and not surfaced by the returned
    regions matrix, so this fixture cannot directly check for it.
    """
    chm = np.array([
        [0, 0,  0, 0,  0],
        [0, 10, 6, 10, 0],
        [0, 0,  0, 0,  0],
        [0, 0,  0, 0,  0],
        [0, 0,  0, 0,  0],
    ], dtype=np.float64)
    seeds = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int32)
    return {
        "inputs/chm": chm,
        "inputs/seeds": seeds,
        "inputs/th_tree": np.float64(2.0),
        "inputs/th_seed": np.float64(0.45),
        "inputs/th_cr":   np.float64(0.55),
        "inputs/max_cr":  np.float64(10.0),
        "expected/regions": expected,
        "meta/source": "manual_derivation: lidR/src/C_dalponte2016.cpp:73-123",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Two seeds equidistant from neighbour (1,2). Row-major loop "
            "order ⇒ seed2 (later in the row) overwrites seed1 in Regiontemp."
        ),
    }


# --- silva2016 fixtures ------------------------------------------------------

def fx_silva2016_happy() -> dict:
    """Two trees, ten points: 4 each near the trees plus one too-far and one
    too-low. Verifies 1-based id mapping and both filter axes."""
    xyz = np.array([
        [0.0,  0.0,  10.0],
        [1.0,  0.0,   8.0],
        [0.0,  1.0,   8.0],
        [-1.0, 0.0,   7.0],
        [10.0, 10.0, 10.0],
        [11.0, 10.0,  8.0],
        [10.0, 11.0,  8.0],
        [9.0,  10.0,  7.0],
        [-7.0, 0.0,   8.0],   # nearest tree1, d=7 > 6 = 0.6*10 → drop
        [0.5,  0.0,   2.0],   # nearest tree1, z=2 < 3 = 0.3*10 → drop
    ], dtype=np.float64)
    treetops = np.array([
        [0.0,  0.0,  10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float64)
    expected = np.array([1, 1, 1, 1, 2, 2, 2, 2, 0, 0], dtype=np.int32)
    return {
        "inputs/xyz": xyz,
        "inputs/treetops": treetops,
        "inputs/max_cr_factor": np.float64(0.6),
        "inputs/exclusion":     np.float64(0.3),
        "expected/ids": expected,
        "meta/source": "manual_derivation: lidR/R/algorithm-its.R:203-279",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Voronoi by xy: hmax/tree=10 ⇒ thresh d≤6, z≥3. The 7-distant "
            "and z=2 points get filtered to 0; the rest map to 1 or 2."
        ),
    }


def fx_silva2016_degenerate_empty_treetops() -> dict:
    """No treetops ⇒ unconditionally all zeros. lidR returns NA_integer_
    (raster, no segmentation); our int32 convention is 0 = unassigned."""
    xyz = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float64)
    treetops = np.zeros((0, 3), dtype=np.float64)
    expected = np.zeros((2,), dtype=np.int32)
    return {
        "inputs/xyz": xyz,
        "inputs/treetops": treetops,
        "inputs/max_cr_factor": np.float64(0.6),
        "inputs/exclusion":     np.float64(0.3),
        "expected/ids": expected,
        "meta/source": "manual_derivation: lidR/R/algorithm-its.R:227-232",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": "Empty treetops short-circuits to all-zero ids.",
    }


def fx_silva2016_corner_boundary() -> dict:
    """Single tree with points hitting the inclusive boundary on both filters.

    hmax=10 ⇒ z-thresh=3.0 (≥), d-thresh=6.0 (≤). Points at exactly z=3 and
    d=6 are kept; just-below-z and just-above-d are dropped. Catches off-by-
    one in the inclusive comparisons.
    """
    xyz = np.array([
        [0.0, 0.0, 10.0],   # the tree itself: d=0, z=10 → keep
        [3.0, 0.0,  3.0],   # boundary on z (z==3.0 ≥ 3.0) → keep
        [6.0, 0.0,  5.0],   # boundary on d (d==6.0 ≤ 6.0) → keep
        [3.0, 0.0,  2.0],   # z=2 < 3 → drop
        [7.0, 0.0,  5.0],   # d=7 > 6 → drop
    ], dtype=np.float64)
    treetops = np.array([[0.0, 0.0, 10.0]], dtype=np.float64)
    expected = np.array([1, 1, 1, 0, 0], dtype=np.int32)
    return {
        "inputs/xyz": xyz,
        "inputs/treetops": treetops,
        "inputs/max_cr_factor": np.float64(0.6),
        "inputs/exclusion":     np.float64(0.3),
        "expected/ids": expected,
        "meta/source": "manual_derivation: lidR/R/algorithm-its.R:268-272",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Inclusive boundary check: z >= exclusion*hmax and "
            "d <= max_cr_factor*hmax. Both keep the equality case."
        ),
    }


FIXTURES = {
    "dalponte2016_happy":       fx_dalponte2016_happy,
    "dalponte2016_degenerate":  fx_dalponte2016_degenerate,
    "dalponte2016_corner_tie":  fx_dalponte2016_corner_tie,
    "silva2016_happy":               fx_silva2016_happy,
    "silva2016_degenerate_empty":    fx_silva2016_degenerate_empty_treetops,
    "silva2016_corner_boundary":     fx_silva2016_corner_boundary,
}


def main() -> None:
    for name, fn in FIXTURES.items():
        payload = fn()
        out_path = HERE / f"{name}.npz"
        np.savez(out_path, **payload)
        size = out_path.stat().st_size
        if size > 50_000:
            raise AssertionError(
                f"{name}.npz exceeds 50KB cap (got {size} bytes); "
                f"shrink the fixture per spec §6."
            )
        print(f"wrote {out_path.name:40s}  ({size:5d} B)")


if __name__ == "__main__":
    main()
