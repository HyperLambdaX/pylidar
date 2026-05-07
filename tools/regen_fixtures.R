# tools/regen_fixtures.R
#
# Skeleton — does nothing in M0.
#
# Purpose (post-M3, when an R + lidR environment is available):
#   Run real lidR against the same inputs encoded in tests/fixtures/*.npz and
#   overwrite the `expected/*` arrays plus `meta/source = "lidR_run"` and
#   `meta/lidR_commit_ref = "<hash>"`. The same pytest suite then becomes a
#   regression gate against the upstream reference implementation.
#
# Until then, fixtures are produced by manual derivation from lidR source
# (spec §6, `meta/source = "manual_derivation"`).
#
# Intentionally empty:

# regen_dalponte2016 <- function(fixture_path) { ... }   # M1+
# regen_silva2016    <- function(fixture_path) { ... }   # M1+
# regen_li2012       <- function(fixture_path) { ... }   # M2+
# regen_lmf          <- function(fixture_path) { ... }   # M2+
# regen_chm_smooth   <- function(fixture_path) { ... }   # M3+
# regen_watershed    <- function(fixture_path) { ... }   # M3+
