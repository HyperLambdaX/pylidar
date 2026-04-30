// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — common point-cloud and raster value types.
//
// PointCloudXYZ: non-owning POD view over 3 coordinate streams. The `stride`
// field counts in DOUBLE ELEMENTS, not bytes, and lets the same struct view
// either an interleaved (N,3) row-major numpy array (stride=3) or three
// separate std::vector<double> buffers (stride=1).
//
// RasterView<T>: owning 2D matrix paired with a (origin, pixel_size) affine
// from raster coordinates to world XY. Only translation + isotropic pixel size
// — no rotation, no shear (spec §6.1).
//
// TreeTop: POD result type for tree-top detection. `id == 0` means "unassigned"
// — the caller is expected to fill in 1..M before passing to algorithms that
// require labelled seeds (dalponte2016, silva2016).

#pragma once

#include <cstddef>
#include <cstdint>

#include "matrix2d.hpp"

namespace pylidar::common {

struct PointCloudXYZ {
    const double* x      = nullptr;
    const double* y      = nullptr;
    const double* z      = nullptr;
    std::size_t   n      = 0;
    std::size_t   stride = 1;  // double elements between consecutive points

    double get_x(std::size_t i) const noexcept { return x[i * stride]; }
    double get_y(std::size_t i) const noexcept { return y[i * stride]; }
    double get_z(std::size_t i) const noexcept { return z[i * stride]; }
};

template <class T>
struct RasterView {
    Matrix2D<T> data;
    double      origin_x   = 0.0;  // world x of pixel (row=0, col=0) center
    double      origin_y   = 0.0;  // world y of pixel (row=0, col=0) center
    double      pixel_size = 1.0;  // edge length, > 0

    // GIS convention: row 0 is the northern edge (largest y); world y decreases
    // as row index increases.
    double world_x(std::size_t col) const noexcept {
        return origin_x + static_cast<double>(col) * pixel_size;
    }
    double world_y(std::size_t row) const noexcept {
        return origin_y - static_cast<double>(row) * pixel_size;
    }
};

struct TreeTop {
    double  x  = 0.0;
    double  y  = 0.0;
    double  z  = 0.0;
    std::int32_t id = 0;  // 0 = unassigned
};

}  // namespace pylidar::common
