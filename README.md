# pylidar

Python port of [lidR](https://github.com/r-lidar/lidR)'s individual tree segmentation
algorithms (dalponte2016, silva2016, li2012, lmf, chm_smooth, watershed). The C++
core is wrapped with nanobind and built via scikit-build-core; algorithms are
exposed under `pylidar.segmentation`. Licensed under GPL-3.0-or-later (inherited
from lidR).

## Install

The project is managed with [`uv`](https://docs.astral.sh/uv/). Pick the mode
that matches your goal.

### Dev mode (editable, recompiles on `.cpp`/`.hpp` edits)

```bash
# 1. Install runtime + dev deps (build backend, ninja, pytest, ...) into .venv
#    via the lockfile, but skip uv's editable install of the project itself.
#    uv's editable mode bakes an ephemeral ninja path into the CMake cache,
#    which breaks scikit-build-core's redirect-mode rebuild hook later.
uv sync --extra dev --no-install-project

# 2. Install the project editable through `uv pip` with --no-build-isolation,
#    so the build re-uses the venv's stable backend + ninja paths and
#    `editable.rebuild = true` can rerun CMake on .cpp/.hpp edits.
uv pip install --no-build-isolation -e ".[dev]"

# 3. Smoke test.
uv run pytest -q
```

After this, editing any `src/**/*.cpp|*.hpp` and re-importing `pylidar._core`
will trigger an automatic incremental rebuild + reinstall of `_core.<abi>.so`.

### Use mode (one-shot install, no rebuild on edits)

```bash
uv venv
uv pip install .
```

This builds a wheel in an ephemeral env and installs the resulting `_core.so`
into the venv. No rebuild hook, no source rebuild on edits — closer to what
end users will eventually get from a published package.

> Note: an unrelated package named `pylidar` already exists on PyPI. This
> project is currently for local install only; the name will be revisited
> before any public release.
