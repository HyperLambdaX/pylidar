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


# --- li2012 fixtures ---------------------------------------------------------

def fx_li2012_happy() -> dict:
    """Two clearly separated trees; verifies tree-id assignment + R>0 LMF
    pre-pass + radius cutoff.

    Hand-trace of `lidR/src/LAS.cpp::segment_trees`:

      Points (sorted descending z): p0(0,0,10), p2(5,0,9), p1(1,0,8),
                                    p3(5,1,7).
      LMF (R=2.0, hws=1, r²=1):
        - p0: only p1 within disc (dist²=1 ≤ 1); p1.z=8<10 ⇒ cascade
              state[1]=NLM; p0 stays LM.
        - p1: state==NLM ⇒ skip, lm[1]=0.
        - p2: only p3 within disc (dist²=1); p3.z=7<9 ⇒ cascade NLM[3];
              p2 stays LM.
        - p3: NLM, skip.
        ⇒ is_lm = [1, 0, 1, 0].
      dummy = (xmin-100, ymin-100, 0) = (-100, -100, 0).
      Iter 1, u=p0 (z=10≥hmin=2):
        - i=1 (p2 in U order): d_to_u=25 ≤ radius²=100. dmin1=25 (only p0
          in P). dmin2=21025 (dummy). dt=2.25 (z=9≤Zu=15). is_lm=1.
          dmin1>dt ⇒ N. inN[1]=1.
        - i=2 (p1): d_to_u=1. dmin1=1. dmin2=min(20201, sqd(p1,p2)=16)=16.
          dt=2.25. is_lm=0 (cascade). dmin1<=dmin2 ⇒ P. ids[1]=1.
        - i=3 (p3): d_to_u=26. dmin1=min(sqd(p3,p0)=26, sqd(p3,p1)=17)=17.
          dmin2=min(sqd(p3,dummy)=21226, sqd(p3,p2)=1)=1. is_lm=0.
          dmin1>dmin2 ⇒ N. inN[3]=1.
      End iter1: ids=[1,1,0,0]. next_U=[p2,p3]. k=2.
      Iter 2, u=p2 (z=9):
        - i=1 (p3): d_to_u=1. dmin1=1. dmin2=21226. is_lm=0. ⇒ P. ids[3]=2.
      End iter2: ids=[1,1,2,2]. ✓
    """
    xyz = np.array([
        [0.0, 0.0, 10.0],   # p0 — peak A
        [1.0, 0.0,  8.0],   # p1 — under A
        [5.0, 0.0,  9.0],   # p2 — peak B
        [5.0, 1.0,  7.0],   # p3 — under B
    ], dtype=np.float64)
    expected = np.array([1, 1, 2, 2], dtype=np.int32)
    return {
        "inputs/xyz":       xyz,
        "inputs/dt1":       np.float64(1.5),
        "inputs/dt2":       np.float64(2.0),
        "inputs/Zu":        np.float64(15.0),
        "inputs/R":         np.float64(2.0),
        "inputs/hmin":      np.float64(2.0),
        "inputs/speed_up":  np.float64(10.0),
        "expected/ids":     expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:1113-1280 + LAS.cpp:399-480",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Two well-separated peaks; LMF pre-pass collapses each cluster "
            "to one LM seed; segmentation produces tree ids 1 and 2."
        ),
    }


def fx_li2012_degenerate_below_hmin() -> dict:
    """All points below hmin ⇒ outer loop bails out at first iteration; no
    trees formed. lidR's no-op iterate-and-empty produces same result.

    Trace: U sorted descending = [p0(z=1.5), p1(1.0), p2(0.5)]. Iter 1:
    u=p0, u.z=1.5 < th_tree=hmin=2.0 ⇒ break. ids stays all zero.
    """
    xyz = np.array([
        [0.0, 0.0, 1.5],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 0.5],
    ], dtype=np.float64)
    expected = np.zeros((3,), dtype=np.int32)
    return {
        "inputs/xyz":       xyz,
        "inputs/dt1":       np.float64(1.5),
        "inputs/dt2":       np.float64(2.0),
        "inputs/Zu":        np.float64(15.0),
        "inputs/R":         np.float64(2.0),
        "inputs/hmin":      np.float64(2.0),
        "inputs/speed_up":  np.float64(10.0),
        "expected/ids":     expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:1178-1181",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Highest remaining point below th_tree=hmin ⇒ no tree forms. "
            "Output stays all-zero (our 'unassigned' convention)."
        ),
    }


def fx_li2012_corner_radius_clip() -> dict:
    """Locks the fix for the M2-review HIGH bug: when the radius gate
    fires (`d_to_u > radius²`), lidR/LAS.cpp:1216-1219 only sets
    ``inN[i] = true`` and does **not** push the point into N. An earlier
    pylidar revision pushed into ``N_`` here, polluting later ``dmin2``
    lookups and flipping non-LM-rule verdicts on borderline points.

    Trace (R=20 ⇒ hws=10, r²=100; radius=10, radius²=100; dt1=1.5,
    dt1²=2.25; Zu=15, dt1 used; hmin=2):

      Points: u=(0,0,10), pf=(15,0,9), pn=(8,0,7).
      LMF pre-pass (hws=10, r²=100 + EPSILON):
        - u: pf at d²=225 > 100 (out). pn at d²=64 ≤ 100 (in); pn.z<u.z
             ⇒ cascade NLM[pn]. is_lm[u]=1.
        - pf: u at d²=225 (out). pn at d²=49 ≤ 100 (in); pn already NLM
              cascaded; pn.z<pf.z ⇒ NLM (no-op). is_lm[pf]=1.
        - pn: state NLM ⇒ skip, is_lm[pn]=0.
        ⇒ is_lm = [1, 1, 0].
      dummy = (xmin-100, ymin-100, 0) = (-100, -100, 0).
      Iter 1, u_top=u (z=10 ≥ hmin):
        - i=1 (pf): d_to_u=225 > 100 ⇒ radius gate. inN[1]=1, **pf NOT
          pushed to N_** (this is the lidR-correct behaviour).
        - i=2 (pn): d_to_u=64 ≤ 100 ⇒ rules. dmin1=64. dmin2=sqd(pn,
          dummy)=21664 (N_ is empty). dt=2.25. is_lm[pn]=0. dmin1≤dmin2?
          64≤21664 yes ⇒ P. ids[pn]=1.
      End iter1: ids=[1,0,1]. next_U=[pf]. (pn NOT in next_U because pn
      went to P, not N — verified by the LM-rule branch above.)
      Iter 2, u_top=pf (z=9): no further candidates ⇒ ids[pf]=2.
      Final ids = [1, 2, 1].

    With the pre-fix code (which wrongly pushed pf to N_ on the gate),
    pn's dmin2 in iter 1 collapses to sqd(pn,pf)=49 < dmin1=64, the
    non-LM rule flips ⇒ pn goes to N ⇒ pn ends up in pf's tree as id=2.
    Output diverges to [1, 2, 2]. This fixture pins the corrected output.
    """
    xyz = np.array([
        [0.0,  0.0, 10.0],
        [15.0, 0.0,  9.0],
        [8.0,  0.0,  7.0],
    ], dtype=np.float64)
    expected = np.array([1, 2, 1], dtype=np.int32)
    return {
        "inputs/xyz":       xyz,
        "inputs/dt1":       np.float64(1.5),
        "inputs/dt2":       np.float64(2.0),
        "inputs/Zu":        np.float64(15.0),
        "inputs/R":         np.float64(20.0),
        "inputs/hmin":      np.float64(2.0),
        "inputs/speed_up":  np.float64(10.0),
        "expected/ids":     expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:1216-1219 (radius gate)",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Radius-gate regression lock. lidR sets only inN[i] when "
            "d>radius; pushing into N would flip pn's tree assignment."
        ),
    }


def fx_li2012_corner_R0_skips_prepass() -> dict:
    """R=0 takes the `is_lm.assign(ni, 1)` branch; output differs from R>0
    when a follower would be cascaded NLM at R>0 but kept LM at R=0, and
    the LM-rule rejection then forces a separate tree.

    Trace (R=0 ⇒ is_lm = all 1; dt1=1.5 ⇒ dt1²=2.25; Zu=15; hmin=2):

      Points: p0=(0,0,10), p1=(1.6,0,8). xmin=0, ymin=0, dummy=(-100,-100,0).
      U sorted descending by z = [p0, p1].
      Iter 1, u=p0 (z=10):
        - i=1 (p1): d_to_u=2.56 ≤ radius²=100. dmin1=2.56.
          dmin2=sqd(p1,dummy)=10401+10000=20401 — wait, (1.6+100)²+
          (0+100)² = 10322.56+10000 = 20322.56. dt=2.25. is_lm[p1]=1.
          dmin1>dt? 2.56>2.25 YES ⇒ N. inN[1]=1, N_=[p1].
      End iter1: ids=[1, 0]. next_U=[p1]. k=2.
      Iter 2, u=p1 (z=8 ≥ hmin): ids[p1]=2.
      Final ids = [1, 2]. ✓

    Compared with the default R=2 (hws=1, r²=1): p1 at d²=2.56 > 1 ⇒
    NOT in p0's disc, no cascade ⇒ same is_lm=[1,1] ⇒ same path ⇒ same
    output [1, 2]. So R doesn't differentiate at this geometry — but
    R=0 vs R>0 with a closer follower would (R≥3.2 enough): kept here as
    a coverage stamp on the `R<=0` branch in li2012.cpp:140-144.
    """
    xyz = np.array([
        [0.0, 0.0, 10.0],
        [1.6, 0.0,  8.0],
    ], dtype=np.float64)
    expected = np.array([1, 2], dtype=np.int32)
    return {
        "inputs/xyz":       xyz,
        "inputs/dt1":       np.float64(1.5),
        "inputs/dt2":       np.float64(2.0),
        "inputs/Zu":        np.float64(15.0),
        "inputs/R":         np.float64(0.0),
        "inputs/hmin":      np.float64(2.0),
        "inputs/speed_up":  np.float64(10.0),
        "expected/ids":     expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:1140-1150",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "R=0 ⇒ skip LMF pre-pass (every point counts as LM). Output "
            "[1,2] follows from dmin1=2.56 > dt1²=2.25 ⇒ p1 to N ⇒ "
            "becomes its own tree."
        ),
    }


def fx_li2012_corner_th_boundary() -> dict:
    """Top point at exactly hmin still forms a tree (lidR uses strict `<`,
    not `<=`); follower below hmin is included in its parent tree
    (per-point hmin only gates outer-loop entry, not in-tree assignment).

    Trace: p0(0,0,2.0), p1(0.5,0,1.5). U = [p0, p1].
      LMF (R=2, hws=1, r²=1): p0 vs p1 dist²=0.25 ≤ 1; p0.z>p1.z ⇒
        cascade state[1]=NLM. is_lm = [1, 0].
      Iter 1, u=p0 (z=2.0). 2.0 < 2.0 == false ⇒ enter. ids[0]=1.
        - i=1 (p1): d_to_u=0.25. dmin1=0.25. dmin2≈sqd(p1,dummy)=20100.25.
          dt=2.25 (z=1.5≤15). is_lm=0. dmin1<=dmin2 ⇒ P. ids[1]=1.
      Final ids = [1, 1]. ✓
    """
    xyz = np.array([
        [0.0, 0.0, 2.0],
        [0.5, 0.0, 1.5],
    ], dtype=np.float64)
    expected = np.array([1, 1], dtype=np.int32)
    return {
        "inputs/xyz":       xyz,
        "inputs/dt1":       np.float64(1.5),
        "inputs/dt2":       np.float64(2.0),
        "inputs/Zu":        np.float64(15.0),
        "inputs/R":         np.float64(2.0),
        "inputs/hmin":      np.float64(2.0),
        "inputs/speed_up":  np.float64(10.0),
        "expected/ids":     expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:1178 (strict <)",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "u.z exactly at hmin: lidR's `<` test is strict so the tree "
            "still forms; below-hmin follower joins it via the non-LM rule."
        ),
    }


# --- lmf_points fixtures -----------------------------------------------------

def fx_lmf_points_happy() -> dict:
    """Two clean peaks with three followers each. ws=2.5 circular.

    Trace (hws=1.25, r²=1.5625):
      p0(0,0,10): p1(1,0,8) at dist²=1 ≤ 1.5625, z<10 ⇒ cascade NLM[1].
                  p2(0,1,7) at dist²=1, z<10 ⇒ NLM[2]. is_max true ⇒ LM.
      p1: state NLM ⇒ skip, lm=0.
      p2: state NLM ⇒ skip, lm=0.
      p3(5,5,9): p4(5.5,5,7) at dist²=0.25, z<9 ⇒ NLM[4]. LM.
      p4: state NLM ⇒ skip, lm=0.
      ⇒ lm = [1, 0, 0, 1, 0].
    """
    xyz = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 0.0,  8.0],
        [0.0, 1.0,  7.0],
        [5.0, 5.0,  9.0],
        [5.5, 5.0,  7.0],
    ], dtype=np.float64)
    expected = np.array([True, False, False, True, False], dtype=np.bool_)
    return {
        "inputs/xyz":       xyz,
        "inputs/ws":        np.float64(2.5),    # scalar — Python wrapper expands
        "inputs/hmin":      np.float64(2.0),
        "inputs/shape":     "circular",
        "expected/lm":      expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:399-480",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Scalar ws ⇒ is_uniform=true ⇒ cascading-NLM optimisation "
            "fires. Two peaks isolated by ≥5 in xy."
        ),
    }


def fx_lmf_points_degenerate_below_hmin() -> dict:
    """Every point z < hmin ⇒ pre-pass marks all NLM ⇒ no LMs."""
    xyz = np.array([
        [0.0, 0.0, 1.5],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 0.5],
    ], dtype=np.float64)
    expected = np.zeros((3,), dtype=np.bool_)
    return {
        "inputs/xyz":       xyz,
        "inputs/ws":        np.float64(2.5),
        "inputs/hmin":      np.float64(2.0),
        "inputs/shape":     "circular",
        "expected/lm":      expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:413",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": "All z<hmin ⇒ all pre-marked NLM, lm = all-false.",
    }


def fx_lmf_points_corner_square_shape() -> dict:
    """Square shape ws=2 includes the box corner at offset (1,1) which
    circular ws=2 (hws=1, r²=1) excludes (corner d²=2 > 1). Locks the
    square post-filter path in lmf.cpp.

    Points: p0=(0,0,10), p1=(1,1,8), p2=(2,0,7). hmin=2, ws=2 square,
    hws=1 ⇒ box [-1,1]×[-1,1] (lidR Rectangle::contains tolerance
    matches via EPSILON).
      p0: p1 at |dx|=1,|dy|=1 ⇒ in box. p1.z<10 ⇒ cascade NLM[1].
          p2 at |dx|=2 > 1+EPSILON ⇒ NOT in box. is_lm[0]=1.
      p1: state NLM ⇒ skip. lm[1]=0.
      p2: window box [|dx|≤1, |dy|≤1] from (2,0). p1 at |dx|=1,|dy|=1
          in box. p1.z=8 > p2.z=7 ⇒ is_max=false. p0 at |dx|=2 > 1
          ⇒ not in box. is_max stays false; is_uniform=true so cascade
          would NLM p1 if zj<zi but zj>zi here ⇒ no cascade. lm[2]=0.
      ⇒ lm = [1, 0, 0].
    """
    xyz = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 1.0,  8.0],
        [2.0, 0.0,  7.0],
    ], dtype=np.float64)
    expected = np.array([True, False, False], dtype=np.bool_)
    return {
        "inputs/xyz":       xyz,
        "inputs/ws":        np.float64(2.0),
        "inputs/hmin":      np.float64(2.0),
        "inputs/shape":     "square",
        "expected/lm":      expected,
        "meta/source": "manual_derivation: lidR/inst/include/lidR/Shapes.h:71-86 (Rectangle)",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Square box includes (1,1) corner; circular disc r²=1 would "
            "exclude it. Pins the lmf.cpp square post-filter."
        ),
    }


def fx_lmf_points_corner_nonuniform_ws_cascade_off() -> dict:
    """Non-uniform ws disables cascading-NLM. With cascade ON (the
    incorrect behaviour for non-uniform), p_small would be wrongly marked
    NLM by p_big's iteration; with cascade OFF (correct), p_small's own
    iteration sees no neighbours within its tiny disc and stays LM.

    Points: p_big=(0,0,10), p_small=(3,0,9).
      ws[big]=10 (hws=5), ws[small]=1 (hws=0.5). hmin=2, circular.
      Cascade-OFF (is_uniform=false ⇒ this fixture's path):
        - p_big: hws=5, p_small at d=3 ≤ 5+EPSILON ⇒ in disc. zj<zi.
          Cascade gated off ⇒ state[small] stays UKN. is_lm[big]=1.
        - p_small: state UKN. hws=0.5, p_big at d=3 > 0.5+EPSILON ⇒
          NOT in disc. is_max=true. is_lm[small]=1.
        ⇒ lm = [True, True].
      Cascade-ON (is_uniform=true) would produce lm=[True, False] —
      hence the test asserts on the cascade-off output and indirectly
      on the gating logic.
    """
    xyz = np.array([
        [0.0, 0.0, 10.0],
        [3.0, 0.0,  9.0],
    ], dtype=np.float64)
    ws_arr = np.array([10.0, 1.0], dtype=np.float64)
    expected = np.array([True, True], dtype=np.bool_)
    return {
        "inputs/xyz":       xyz,
        "inputs/ws":        ws_arr,
        "inputs/hmin":      np.float64(2.0),
        "inputs/shape":     "circular",
        "expected/lm":      expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:457 (vws gate)",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Per-point ws array ⇒ is_uniform=false ⇒ cascade gated off. "
            "Output [True,True] only stays correct because cascade is off."
        ),
    }


def fx_lmf_points_corner_tie_equal_z() -> dict:
    """Two equal-z points within window: row-major first wins.

    Trace: p0(0,0,10), p1(1,0,10). ws=2.5, hws=1.25, r²=1.5625.
      p0 (i=0): p1 within disc (dist²=1). zj==zi but lm[1]=0 (not yet set)
                ⇒ tie-break does NOT fire. is_max true ⇒ lm[0]=1.
      p1 (i=1): p0 within disc. zj==zi and lm[0]=1 ⇒ tie-break fires
                ⇒ is_max=false ⇒ lm[1]=0.
      ⇒ lm = [1, 0]. (cf. lidR's parallel version is order-dependent;
       sequential row-major is one valid serialisation.)
    """
    xyz = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
    ], dtype=np.float64)
    expected = np.array([True, False], dtype=np.bool_)
    return {
        "inputs/xyz":       xyz,
        "inputs/ws":        np.float64(2.5),
        "inputs/hmin":      np.float64(2.0),
        "inputs/shape":     "circular",
        "expected/lm":      expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:464-468",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Equal-z neighbours within disc ⇒ first iteration in row order "
            "claims LM; second loses via the filter[pt.id] tie-break."
        ),
    }


# --- lmf_chm fixtures --------------------------------------------------------

def fx_lmf_chm_happy() -> dict:
    """5x5 CHM with two clean peaks; ws=3 circular ⇒ both detected."""
    chm = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 6.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    # All zero cells fall below hmin=2.0 ⇒ pre-pass marks them NLM. The
    # only candidates the main loop visits are (1,1) and (3,3); both have
    # only zero neighbours so each is trivially a local max.
    expected = np.array([[1, 1], [3, 3]], dtype=np.int32)
    return {
        "inputs/chm":   chm,
        "inputs/ws":    np.float64(3.0),
        "inputs/hmin":  np.float64(2.0),
        "inputs/shape": "circular",
        "expected/coords": expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:399-480 (raster path)",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": "Two isolated peaks at (1,1) and (3,3); 3x3 windows.",
    }


def fx_lmf_chm_degenerate_all_below_hmin() -> dict:
    """3x3 CHM all zeros — every cell below hmin ⇒ no LMs ⇒ (0,2) output."""
    chm = np.zeros((3, 3), dtype=np.float64)
    expected = np.zeros((0, 2), dtype=np.int32)
    return {
        "inputs/chm":   chm,
        "inputs/ws":    np.float64(3.0),
        "inputs/hmin":  np.float64(2.0),
        "inputs/shape": "circular",
        "expected/coords": expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:413",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": "All cells z<hmin ⇒ pre-pass NLM, no LM cells emitted.",
    }


def fx_lmf_chm_corner_square_shape() -> dict:
    """Square shape ws=2 (hws=1, hws_int=1, integer box [-1,1]×[-1,1])
    includes the (1,1) corner that circular shape excludes (disc r²=1
    rejects dr²+dc²=2). Layout chosen so the difference matters: the
    secondary peak at (1,1) is hidden by (0,0) under SQUARE (cascade
    NLM via box neighbour) but visible under CIRCULAR (the (0,0) peak
    is outside (1,1)'s disc).

    Trace (ws=2, hws=1, hws_int=1, hws²=1):
      Layout:
        5 0 0
        0 4 0
        0 0 0
      hmin=2 ⇒ pre-NLM all zero cells. Candidates: (0,0)=5, (1,1)=4.
      Visit (0,0): box [0..1, 0..1]. Square neighbours:
        - (0,1)=0  cascade NLM (already).
        - (1,0)=0  cascade NLM.
        - (1,1)=4  zj<zi=5 ⇒ cascade NLM[(1,1)] ← this is what differs
          from circular (where (1,1) is at d²=2 > 1, not a neighbour).
        is_max=true ⇒ emit (0,0). lm_grid[(0,0)]=1.
      Visit (1,1): state NLM (cascaded above) ⇒ skip.
      ⇒ coords = [[0, 0]].

    Under circular shape, (1,1) would also be a LM (its window doesn't
    reach (0,0)) and the output would be [[0,0],[1,1]] — the asymmetry
    pins the square branch.
    """
    chm = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)
    expected = np.array([[0, 0]], dtype=np.int32)
    return {
        "inputs/chm":   chm,
        "inputs/ws":    np.float64(2.0),
        "inputs/hmin":  np.float64(2.0),
        "inputs/shape": "square",
        "expected/coords": expected,
        "meta/source": "manual_derivation: lidR/inst/include/lidR/Shapes.h:71-86 (Rectangle)",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Square ws=2 includes the (1,1) box corner that circular "
            "ws=2 excludes (d²=2 > r²=1). Cascades the secondary peak."
        ),
    }


def fx_lmf_chm_corner_tie_equal_neighbors() -> dict:
    """Two equal-z adjacent cells: row-major scan ⇒ left wins.

    Trace (3x3, ws=3 ⇒ hws=1.5, hws_int=1, hws²=2.25):
      Cells (0,*) all zero ⇒ pre-NLM. (1,0)=0 ⇒ pre-NLM. (2,*) all zero
      ⇒ pre-NLM. Candidates: (1,1)=5, (1,2)=5.
      (1,1) iterates first: window [0..2, 0..2]. Neighbour (1,2)=5 within
      disc (dist²=1 ≤ 2.25); zj==zi but lm_grid[(1,2)]=0 (not yet set) ⇒
      tie-break does NOT fire. All other neighbours are zero. is_max true
      ⇒ emit (1,1).
      (1,2) iterates next: window [0..2, 1..2] (right edge clipped).
      Neighbour (1,1)=5 within disc; lm_grid[(1,1)]=1 ⇒ tie-break fires
      ⇒ return false. Not emitted.
      ⇒ coords = [[1, 1]].
    """
    chm = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 5.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)
    expected = np.array([[1, 1]], dtype=np.int32)
    return {
        "inputs/chm":   chm,
        "inputs/ws":    np.float64(3.0),
        "inputs/hmin":  np.float64(2.0),
        "inputs/shape": "circular",
        "expected/coords": expected,
        "meta/source": "manual_derivation: lidR/src/LAS.cpp:464-468",
        "meta/lidR_commit_ref": "TBD: regen with R env",
        "meta/notes": (
            "Row-major scan tie-break: left equal-z neighbour claims LM, "
            "right loses via lm_grid[earlier]==1 lookup."
        ),
    }


FIXTURES = {
    "dalponte2016_happy":       fx_dalponte2016_happy,
    "dalponte2016_degenerate":  fx_dalponte2016_degenerate,
    "dalponte2016_corner_tie":  fx_dalponte2016_corner_tie,
    "silva2016_happy":               fx_silva2016_happy,
    "silva2016_degenerate_empty":    fx_silva2016_degenerate_empty_treetops,
    "silva2016_corner_boundary":     fx_silva2016_corner_boundary,
    "li2012_happy":                       fx_li2012_happy,
    "li2012_degenerate_below_hmin":       fx_li2012_degenerate_below_hmin,
    "li2012_corner_th_boundary":          fx_li2012_corner_th_boundary,
    "li2012_corner_radius_clip":          fx_li2012_corner_radius_clip,
    "li2012_corner_R0_skips_prepass":     fx_li2012_corner_R0_skips_prepass,
    "lmf_points_happy":                          fx_lmf_points_happy,
    "lmf_points_degenerate_below_hmin":          fx_lmf_points_degenerate_below_hmin,
    "lmf_points_corner_tie_equal_z":             fx_lmf_points_corner_tie_equal_z,
    "lmf_points_corner_square_shape":            fx_lmf_points_corner_square_shape,
    "lmf_points_corner_nonuniform_ws_cascade_off":  fx_lmf_points_corner_nonuniform_ws_cascade_off,
    "lmf_chm_happy":                              fx_lmf_chm_happy,
    "lmf_chm_degenerate_all_below_hmin":          fx_lmf_chm_degenerate_all_below_hmin,
    "lmf_chm_corner_tie_equal_neighbors":         fx_lmf_chm_corner_tie_equal_neighbors,
    "lmf_chm_corner_square_shape":                fx_lmf_chm_corner_square_shape,
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
