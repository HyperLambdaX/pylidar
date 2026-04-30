# pylidar

Individual tree segmentation (ITS) algorithms from the R package
[`lidR`](https://github.com/r-lidar/lidR), ported to a standalone C++17 core
with Python bindings. **Status: pre-release skeleton (v0.1, Phase 0).** No
algorithms are wired up yet — the wheel currently exposes only a logging hook
to validate the build chain.

## Scope

- Algorithm core: pure C++17 (`src/core/`), zero R/Rcpp/Python dependencies,
  organised by algorithm family (`core/its/`, future `core/dtm/`, …).
- Python bindings: [nanobind](https://github.com/wjakob/nanobind) ≥ 2.0 →
  `pylidar._core`.
- Build: [scikit-build-core](https://scikit-build-core.readthedocs.io) +
  [cibuildwheel](https://cibuildwheel.readthedocs.io) for multi-platform wheels
  (Linux x86_64/aarch64, macOS x86_64/arm64, Windows AMD64; Python 3.10–3.14).
- LAS/LAZ I/O is **out of scope** — pylidar accepts numpy arrays only. Use
  [`laspy`](https://github.com/laspy/laspy) on the user side.

The full architecture spec lives at
[`docs/specs/2026-04-30-pylidar-its-design.md`](docs/specs/2026-04-30-pylidar-its-design.md);
the implementation roadmap is in [`task_plan.md`](task_plan.md).

## License

GPL-3.0-or-later, inherited from upstream lidR. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE) for third-party attributions (notably vendored
[`nanoflann`](https://github.com/jlblancoc/nanoflann), BSD-2). Commercial users
who need a non-GPL licence should contact the maintainers.

## Local dev quickstart

Requires CMake ≥ 3.26 and a C++17 compiler with OpenMP. On macOS,
`brew install libomp` first.

```sh
# First-time setup (creates .venv, installs the editable wheel + test deps).
uv venv --python 3.14
uv pip install -e ".[test]"

# Run the test suite — no `source .venv/bin/activate` needed; uv run picks
# up the project's environment automatically.
uv run pytest tests -m "not requires_fixture"
```

After modifying any C++ source you need to rebuild the extension:

```sh
uv pip install -e ".[test]" --no-deps   # rebuilds via scikit-build-core (~3s)
uv run pytest tests -m "not requires_fixture"
```

`uv run` does not trigger CMake — it only resolves and runs the command in
the project env. C++ rebuilds always go through `uv pip install -e`.
