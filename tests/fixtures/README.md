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

## Inventory (M1)

| File | Algorithm | Class |
|---|---|---|
| `dalponte2016_happy.npz`       | dalponte2016 | happy path: two crowns grow on a 5×5 CHM |
| `dalponte2016_degenerate.npz`  | dalponte2016 | degenerate: th_cr gates every neighbour |
| `dalponte2016_corner_tie.npz`  | dalponte2016 | corner: two seeds tie on a shared neighbour |
| `silva2016_happy.npz`          | silva2016    | happy path: 2 trees, both filter axes hit |
| `silva2016_degenerate_empty.npz` | silva2016  | degenerate: empty treetops ⇒ all zeros |
| `silva2016_corner_boundary.npz`  | silva2016  | corner: inclusive boundary on z and d |
