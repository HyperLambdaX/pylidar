// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::lmf (local maximum filter for tree-top detection)
//
// Source: lidR v4.3.2 LAS::filter_local_maxima(NumericVector ws, double
// min_height, bool circular) (src/LAS.cpp:399-480), the variant that
// `C_lmf` (RcppFunction.cpp:37) calls. The "raster path" of upstream lmf is
// not a separate algorithm: lidR converts a CHM to a fake 1-point-per-cell
// LAS via raster_as_las() (R/locate_trees.R:106-148) and runs the same
// point-cloud filter. lmf_chm here mirrors that — it builds a virtual
// PointCloudXYZ from non-NaN cells and dispatches to the same internal
// helper as lmf_points.
//
// v0.1 only supports a fixed scalar `ws` (per spec §6.3); lidR's per-point
// `vws` branch (used when `ws` in R is a function) is out of scope.
//
// Determinism: the implementation diverges from upstream lidR to remove a
// data race in the parallel inner loop — see lmf.cpp file header for the
// two-line summary. User-visible: equal-height neighbours always resolve
// to the lowest-index point as the LM, regardless of thread count.

#pragma once

#include <vector>

#include "common/point_cloud.hpp"
#include "its/shape.hpp"

namespace pylidar::its {

// Local maximum filter on an unstructured point cloud. Returns a list of
// tree tops with `id == 0` — the caller (Python wrapper or downstream
// algorithm) is expected to assign 1..M before passing seeds to dalponte
// 2016 / silva 2016.
//
// Throws std::invalid_argument on ws <= 0 (or non-finite). hmin is allowed
// to be any finite value, including negative (= "no height filter").
std::vector<common::TreeTop> lmf_points(
    const common::PointCloudXYZ& pts,
    double                       ws,
    double                       hmin,
    Shape                        shape);

// Local maximum filter on a CHM raster. Internally rasterises the CHM into
// a fake point cloud (one point per non-NaN cell, world XY from the raster
// transform, z = pixel value) and runs the same filter as lmf_points.
//
// Throws std::invalid_argument on ws <= 0, on chm.pixel_size <= 0, or on
// an empty matrix.
std::vector<common::TreeTop> lmf_chm(
    const common::RasterView<double>& chm,
    double                            ws,
    double                            hmin,
    Shape                             shape);

}  // namespace pylidar::its
