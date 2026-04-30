// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::silva2016 (Voronoi-tessellation tree-crown segmentation)
//
// Port of lidR v4.3.2 silva2016 (R/algorithm-its.R:203-283). Unlike the
// other ITS algorithms, silva2016 is a *pure-R* implementation upstream;
// this file is the first time it's been written in C++. The translation
// trace is in docs/notes/silva2016-translation-trace.md (gitignored,
// user-reviewed mid-phase).
//
// Algorithm in one sentence: for every non-NaN CHM cell, find the nearest
// seed by world-XY Euclidean distance; group cells by their nearest seed
// and compute hmax = max(Z) per group (the *Voronoi-cell* max, NOT the
// seed's own z); a cell is labelled with its seed's id iff
//   Z >= exclusion * hmax  AND  dist <= max_cr_factor * hmax.
//
// The algorithm does no region-growing: cells are independent once the
// per-group hmax is known, so the implementation is three passes over
// the raster (parallel KNN → serial hmax accumulation → parallel write).

#pragma once

#include <cstdint>
#include <vector>

#include "common/matrix2d.hpp"
#include "common/point_cloud.hpp"

namespace pylidar::its {

// Voronoi-tessellation tree-crown segmentation.
//
//   chm           : (H, W) raster + affine transform. NaN cells are
//                   skipped (output stays 0, never enter the Voronoi).
//   seeds         : list of tree tops with caller-assigned `id != 0`.
//                   World XY drives the nearest-neighbour grouping; the
//                   z column is *not* read by the algorithm. Seeds with
//                   id == 0 are skipped. Seeds outside the chm bbox
//                   (including a 0.5-pixel half-skirt around it, matching
//                   the `sf::st_crop` semantics of lidR's
//                   crop_special_its) are silently dropped.
//   max_cr_factor : crown-radius cap as a fraction of hmax. A cell is
//                   accepted iff `dist_to_seed_xy <= max_cr_factor * hmax`.
//                   Must be > 0 and finite. lidR default 0.6.
//   exclusion     : height-threshold lower bound as a fraction of hmax. A
//                   cell is accepted iff `Z >= exclusion * hmax`. Must be
//                   in the OPEN interval (0, 1). lidR default 0.3. Note
//                   the open interval — dalponte uses `[0, 1]` closed for
//                   th_seed/th_cr; silva is stricter upstream.
//
// Returns (H, W) int32 with 0 = unlabelled (NaN cell, no seeds, or
// thresholds failed) and otherwise the seed id.
//
// Throws std::invalid_argument on empty chm, non-finite/non-positive
// pixel_size, max_cr_factor not finite-positive, exclusion outside the
// open interval (0, 1), or non-finite seed XY (callers should validate
// up front but the core self-checks for direct linkage).
common::Matrix2D<std::int32_t> silva2016(
    const common::RasterView<double>&    chm,
    const std::vector<common::TreeTop>&  seeds,
    double                               max_cr_factor,
    double                               exclusion);

}  // namespace pylidar::its
