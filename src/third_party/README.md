# Vendored third-party libraries

Header-only / single-file dependencies copied verbatim into the tree. Each
upstream's original license is preserved alongside its source.

| Library | Version | Path | License | Used by |
|---|---|---|---|---|
| [nanoflann](https://github.com/jlblancoc/nanoflann) | 1.7.1 (`NANOFLANN_VERSION 0x171`) | `nanoflann/nanoflann.h` | BSD-2 (`nanoflann/LICENSE`) | `src/core/common/kdtree.hpp` (M2 onwards) |

## Updating a vendored copy

1. Bump the version in this file.
2. Replace the source file(s) verbatim. Do not patch local edits in.
3. Refresh `LICENSE` if upstream changed it.
4. Rebuild and re-run `uv run pytest -q`.
