// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::smooth_height
//
// Direct port of lidR v4.3.2 LAS::z_smooth (src/LAS.cpp:112-179) with the
// SpatialIndex replaced by a 2D nanoflann tree on PointCloudXYZ.
//
// Phase 1 enum scope (locked with user 2026-04-30): only "mean" and "gaussian"
// methods, matching lidR upstream. The spec's reserved `Median` is deferred to
// v0.2 once a use case appears.
//
// Integer values intentionally match the lidR convention (Square=1, Circular=2;
// Mean=1, Gaussian=2) so future direct ports of LAS.cpp algorithms in this
// family don't need a translation layer.

#pragma once

#include <vector>

#include "common/point_cloud.hpp"
#include "its/shape.hpp"

namespace pylidar::its {

enum class SmoothMethod : int {
    Mean     = 1,
    Gaussian = 2,
};

// Smooth the Z values of a point cloud by averaging (or Gaussian-weighting)
// each point against its 2D-XY neighbours within a `size`-wide window.
//
// Returns a new vector of length pts.n. Throws std::invalid_argument on
// size <= 0 or (Gaussian && sigma <= 0). sigma is unused for Mean.
std::vector<double> smooth_height(
    const common::PointCloudXYZ& pts,
    double                       size,
    SmoothMethod                 method,
    Shape                        shape,
    double                       sigma);

}  // namespace pylidar::its
