# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Greenfield project. As of this writing the repo contains only a `main.py` hello-world entry point, an empty `README.md`, and an empty `dependencies = []` list in `pyproject.toml`. There is no source package, no tests, and no CI yet. Treat structural decisions (package layout, test framework, lint/format tooling) as still open — confirm with the user before introducing them rather than assuming a convention.

## Toolchain

- Python **3.14** is pinned via `.python-version` and `requires-python = ">=3.14"` in `pyproject.toml`. The presence of `.python-version` plus a PEP 621 `pyproject.toml` with no `[build-system]` block indicates this was scaffolded by [`uv`](https://docs.astral.sh/uv/); prefer `uv` commands over raw `pip`/`venv` so the lockfile and interpreter pin stay consistent.
- Run the entry point: `uv run main.py` (or `uv run python main.py`).
- Add a runtime dep: `uv add <pkg>`. Add a dev-only dep: `uv add --dev <pkg>`. Sync after pulling: `uv sync`.
- No test runner is configured yet. If the user asks for tests, ask which framework (pytest is the conventional choice with `uv`) before scaffolding one.

## Project name vs. PyPI

The project is named `pylidar` in `pyproject.toml`. Note that an unrelated package by the same name already exists on PyPI (a SPDLib-based LiDAR processing library). If/when this project gains real functionality and considers publishing, the name will likely need to change — flag this to the user rather than silently working around it.
