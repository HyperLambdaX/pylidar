#pragma once

#include <cstdint>

#include "matrix2d.hpp"

namespace pylidar::core::its {

// Region-growing tree crown segmentation (Dalponte & Coomes 2016).
//
// Inputs:
//   chm     — height raster (rows, cols), row-major; values in meters.
//   seeds   — same shape; non-zero entries are tree IDs at seed pixels,
//             zero entries are unassigned.
//   regions — same shape, mutated in place. Caller must initialise it as a
//             copy of `seeds` before invocation; the algorithm grows IDs
//             outward from each seed via 4-connected propagation.
//   th_tree — pixel height must be strictly greater than this value to grow.
//   th_seed — neighbour height must be strictly greater than
//             th_seed * seed_height to expand.
//   th_crown — neighbour height must be strictly greater than
//              th_crown * mean_crown_height to expand.
//   max_cr  — Chebyshev distance limit (|Δrow|, |Δcol|) from seed (strict <).
//
// Throws std::invalid_argument if the three matrices disagree on shape.
//
// Notes:
//   - Border pixels (row 0, row nrow-1, col 0, col ncol-1) are never used as
//     SOURCE pixels for expansion; they may receive growth as NEIGHBOUR
//     destinations only. This matches lidR's `for (r = 1; r < nrow-1; …)`
//     loop bounds verbatim.
//   - When two seeds expand into the same neighbour pixel within one pass,
//     the later iteration wins (overwrites Region in next-pass view).
//     Bookkeeping (sum_height, npixel) of the losing crown is *not* rolled
//     back — preserved as a 1:1 quirk of the lidR implementation.
void dalponte2016(const Matrix2D<double>& chm,
                  const Matrix2D<int32_t>& seeds,
                  Matrix2D<int32_t>& regions,
                  double th_tree,
                  double th_seed,
                  double th_crown,
                  double max_cr);

}  // namespace pylidar::core::its
