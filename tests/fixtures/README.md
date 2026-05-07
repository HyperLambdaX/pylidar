# tests/fixtures

Each `.npz` here is hand-derived from the upstream lidR source (spec §6
discipline). Inside, every file carries:

- `inputs/<name>` — what the algorithm under test was called with.
- `expected/<name>` — the hand-traced expected output(s).
- `meta/source` — `manual_derivation: lidR/<path>:<line range>` reference.
- `meta/lidR_commit_ref` — `"TBD: regen with R env"` until a real R + lidR
  environment is wired up; then `tools/regen_fixtures.R` will fill it.
- `meta/notes` — short narrative describing what the fixture covers.

## Re-generating

```
uv run python tests/fixtures/_make_fixtures.py
```

The Python generator preserves the hand-trace as code; running it just
serializes the arrays and verifies each file is ≤ 50 KB (spec §6).

## Inventory (M2)

| File | Algorithm | Class |
|---|---|---|
| `dalponte2016_happy.npz`       | dalponte2016 | happy path: two crowns grow on a 5×5 CHM |
| `dalponte2016_degenerate.npz`  | dalponte2016 | degenerate: th_cr gates every neighbour |
| `dalponte2016_corner_tie.npz`  | dalponte2016 | corner: two seeds tie on a shared neighbour |
| `silva2016_happy.npz`          | silva2016    | happy path: 2 trees, both filter axes hit |
| `silva2016_degenerate_empty.npz` | silva2016  | degenerate: empty treetops ⇒ all zeros |
| `silva2016_corner_boundary.npz`  | silva2016  | corner: inclusive boundary on z and d |
| `li2012_happy.npz` | li2012 | happy path: two separated trees |
| `li2012_degenerate_below_hmin.npz` | li2012 | degenerate: all points below hmin |
| `li2012_corner_th_boundary.npz` | li2012 | corner: strict hmin boundary |
| `li2012_corner_radius_clip.npz` | li2012 | corner: radius-gate behaviour |
| `li2012_corner_R0_skips_prepass.npz` | li2012 | corner: R=0 skips LMF pre-pass |
| `lmf_points_happy.npz` | lmf_points | happy path: two point-cloud peaks |
| `lmf_points_degenerate_below_hmin.npz` | lmf_points | degenerate: all points below hmin |
| `lmf_points_corner_tie_equal_z.npz` | lmf_points | corner: equal-z tie-break |
| `lmf_points_corner_square_shape.npz` | lmf_points | corner: square window branch |
| `lmf_points_corner_nonuniform_ws_cascade_off.npz` | lmf_points | corner: non-uniform ws disables cascade |
| `lmf_chm_happy.npz` | lmf_chm | happy path: two CHM peaks |
| `lmf_chm_degenerate_all_below_hmin.npz` | lmf_chm | degenerate: no CHM cells above hmin |
| `lmf_chm_corner_tie_equal_neighbors.npz` | lmf_chm | corner: equal-neighbour tie-break |
| `lmf_chm_corner_square_shape.npz` | lmf_chm | corner: square raster window branch |
