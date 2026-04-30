// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::dalponte2016 (CHM region-growing tree segmentation)
//
// Direct port of lidR v4.3.2 C_dalponte2016 (src/C_dalponte2016.cpp, 126
// lines). The R wrapper (R/algorithm-its.R::dalponte2016) does three jobs
// before calling C: (1) clip the treetops to the CHM bbox, (2) rasterise
// the treetops into an IntegerMatrix where the pixel value = sequential
// seed id, (3) replace NaN cells with -Inf in the canopy. We absorb (2)
// and (3) here so the C++ entry point takes a flat seed list with world
// XY + caller-assigned id, and a CHM that may still contain NaN.
//
// The Python wrapper (pylidar.segment_dalponte2016) handles (1)-style
// out-of-bbox seeds by silently skipping them — same effect as lidR's
// crop_special_its since the seed never lands in a pixel.

#pragma once

#include <cstdint>
#include <vector>

#include "common/matrix2d.hpp"
#include "common/point_cloud.hpp"

namespace pylidar::its {

// Region-grow a CHM crown labelling from a set of seed points.
//
//   chm     : (H, W) raster + affine transform. NaN cells are masked
//             internally (replaced by -inf so the `> th_tree` test
//             rejects them, matching lidR).
//   seeds   : list of tree tops with caller-assigned `id != 0`. World XY
//             is rasterised to the nearest pixel centre via the chm
//             transform; seeds outside the raster are silently dropped.
//             Seeds with id == 0 are skipped (0 is reserved for "no tree").
//   th_seed : neighbour z must exceed `th_seed * h_seed` to be added.
//             Domain [0, 1].
//   th_cr   : neighbour z must exceed `th_cr * mean_crown_z` to be added.
//             Domain [0, 1].
//   th_tree : neighbour z must exceed this absolute height.
//   max_cr  : max pixel distance (Chebyshev, exclusive) from a seed to a
//             grown crown pixel. Must be > 0.
//
// Returns (H, W) int32 matrix where 0 = unlabelled and otherwise the
// seed id.
//
// Throws std::invalid_argument on empty chm, non-finite/non-positive
// pixel_size or max_cr, th_seed/th_cr outside [0, 1], or non-finite
// th_tree.
common::Matrix2D<std::int32_t> dalponte2016(
    const common::RasterView<double>&    chm,
    const std::vector<common::TreeTop>&  seeds,
    double                               th_seed,
    double                               th_cr,
    double                               th_tree,
    double                               max_cr);

}  // namespace pylidar::its
