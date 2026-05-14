#pragma once

#include <cstdint>

#include "matrix2d.hpp"

namespace pylidar::core::its {

// Watershed transform with EBImage semantics ("ext" = neighbourhood half-side,
// "tolerance" = minimum drop from a regional maximum to count as a separate
// basin). 1:1 port of EBImage::watershed (Bioconductor `EBImage/src/watershed.cpp`).
//
// Inputs:
//   chm        — height raster (rows, cols), row-major. Background cells must
//                already be set to 0.0 by the caller (lidR's R wrapper does
//                `Canopy[mask] <- 0` before invoking EBImage); cells <= 0 do
//                NOT receive a label. NaN cells are not supported and must
//                be cleaned upstream — pass NaN-free data.
//   tolerance  — minimum height drop from a regional maximum to seed a
//                separate basin. Hills shallower than this merge into the
//                steepest / closest tolerable neighbour.
//   ext        — neighbourhood half-side (in pixels). ext=1 → 3x3 (8-conn);
//                ext=2 → 5x5; etc. Must be >= 1.
//   out        — same shape as chm; written as int32 labels in 1..K (compact),
//                with 0 marking background / unassigned cells.
//
// Throws std::invalid_argument on shape mismatch, ext < 1, tolerance < 0.
//
// PORT NOTE — porting from EBImage
//   Source:        EBImage src/watershed.cpp (full file) + src/tools.h (PointXY,
//                  POINT_FROM_INDEX, DISTANCE_XY, INDEX_FROM_XY).
//   Layout:        EBImage uses column-major (R native): nx = nrow, index =
//                  x + y*nx where x = row, y = col. We use row-major:
//                  index = r*W + c. The neighbourhood box is symmetric so
//                  geometry is preserved; the only side-effect is plateau
//                  scan order (see Tie-break below).
//   Sort:          EBImage uses R's `rsort_with_index` which is *not* stable
//                  for ties on plateaus → outputs are non-deterministic on
//                  flat regions. We use std::stable_sort tie-broken by
//                  row-major index, making outputs deterministic and
//                  reproducible. As a consequence, our plateau labels may
//                  differ from EBImage's specific (non-deterministic) choice,
//                  but match it bit-exact whenever EBImage is itself
//                  deterministic (no ties).
//   NaN:           EBImage does no NaN handling; lidR's R wrapper pre-cleans.
//                  We mirror — caller must zero NaN cells.
//   Threading:     Single-threaded. EBImage's frame loop is serial; the only
//                  per-image OpenMP would be over `nz` (3rd dim), which we
//                  don't carry (CHM is 2-D).
//   Merging:       check_multiple's "steepest by maxdiff, but override with
//                  closest above tolerance" semantics translated verbatim
//                  (EBImage src/watershed.cpp:151-198).
void watershed_ext(const Matrix2D<double>& chm,
                   double tolerance,
                   int ext,
                   Matrix2D<int32_t>& out);

}  // namespace pylidar::core::its
