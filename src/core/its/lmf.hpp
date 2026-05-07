#pragma once

#include <cstdint>
#include <vector>

#include "matrix2d.hpp"
#include "point.hpp"

namespace pylidar::core::its {

enum class LmfShape {
    Circular,
    Square,
};

// Local-maximum filter on a point cloud (xy-radial neighbour search).
//
// Inputs (xyz):
//   pts        : (N,) PointXYZ
//   ws         : (N,) float64 — full window size (diameter for circular,
//                full side length for square). Bindings layer expands a
//                scalar or callable into this length-N array — core sees
//                only the per-point view.
//   hmin       : exclude points whose z < hmin from being LMs
//   shape      : circular = disc of radius ws[i]/2; square = box with half
//                side ws[i]/2
//   is_uniform : true when bindings layer was passed a scalar ws (all
//                entries equal). Enables the cascading-NLM perf shortcut
//                from lidR/LAS.cpp:457; with non-uniform ws the cascade is
//                disabled because window radii vary
// Output:
//   lm         : (N,) bool (encoded as char 0/1) — true iff pt is a LM.
//
// Behaviour: 1:1 with lidR/src/LAS.cpp:399 `filter_local_maxima(ws, min_height,
// circular)`, sequential. See PORT NOTE in lmf.cpp.
void lmf_points(const std::vector<PointXYZ>& pts,
                const std::vector<double>&  ws,
                double hmin,
                LmfShape shape,
                bool is_uniform,
                std::vector<char>& lm);

// Local-maximum filter on a CHM raster.
//
// Inputs:
//   chm        : (H, W) row-major canopy height (NaN = nodata)
//   ws         : window size in **pixel** units (not world units). For a
//                CHM with non-unit cell size the caller is responsible for
//                converting world-unit ws → pixel ws before calling.
//   hmin       : skip cells with z < hmin
//   shape      : Circular → r ≤ ws/2 in pixel coords; Square → max(|dr|,
//                |dc|) ≤ ws/2
//
// Output:
//   rows / cols: appended (cleared first) with the row/col indices of all
//                detected local maxima, in row-major scan order.
//
// Behaviour: equivalent to running `lmf_points` on a synthetic point cloud
// where each non-NaN cell becomes a point at (col, row, chm[r][c]) — the
// same algorithm lidR's locate_trees(SpatRaster, lmf(...)) takes via
// `raster_as_las`. Implementing it as a direct raster scan (rather than
// materialising the point cloud) saves the kdtree build for the CHM path,
// which is the only intentional optimisation over a literal lidR mirror.
void lmf_chm(const Matrix2D<double>& chm,
             double ws,
             double hmin,
             LmfShape shape,
             std::vector<std::int32_t>& rows,
             std::vector<std::int32_t>& cols);

}  // namespace pylidar::core::its
