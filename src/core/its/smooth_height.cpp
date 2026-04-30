// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::smooth_height implementation. See smooth_height.hpp for
// scope/contract notes; below mirrors lidR LAS::z_smooth line-for-line modulo
// the SpatialIndex → nanoflann substitution.

#include "its/smooth_height.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "common/nanoflann_adaptor.hpp"
#include "common/point_cloud.hpp"

namespace pylidar::its {

std::vector<double> smooth_height(
    const common::PointCloudXYZ& pts,
    double                       size,
    SmoothMethod                 method,
    Shape                        shape,
    double                       sigma)
{
    if (size <= 0.0) {
        throw std::invalid_argument("smooth_height: size must be > 0");
    }
    if (method == SmoothMethod::Gaussian && sigma <= 0.0) {
        throw std::invalid_argument(
            "smooth_height: sigma must be > 0 when method == Gaussian");
    }

    const std::size_t n = pts.n;
    std::vector<double> out(n);
    if (n == 0) return out;

    const double half_res = size / 2.0;
    // Square shape uses a circumscribed-circle radius then filters by bbox.
    const double half_res_sq    = half_res * half_res;
    const double diag_sq        = 2.0 * half_res_sq;
    const double search_rad_sq  = (shape == Shape::Square) ? diag_sq : half_res_sq;

    // Gaussian weight constants. Computed even for Mean (cheap) but unused.
    const double twosquaresigma   = 2.0 * sigma * sigma;
    // M_PI is not portable on MSVC; use std::acos(-1.0).
    const double pi               = std::acos(-1.0);
    const double twosquaresigmapi = twosquaresigma * pi;

    common::PointCloudXYZ_KDAdaptor adaptor(pts);
    common::KDTree2D tree(2, adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    // Signed loop var keeps OpenMP 2.0 (MSVC default) happy if anyone ever
    // builds without /openmp:llvm.
    const std::ptrdiff_t n_signed = static_cast<std::ptrdiff_t>(n);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n_signed; ++i) {
        const std::size_t  i_u = static_cast<std::size_t>(i);
        const double       xi  = pts.get_x(i_u);
        const double       yi  = pts.get_y(i_u);
        const double       query[2] = {xi, yi};

        std::vector<nanoflann::ResultItem<std::uint32_t, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = false;
        tree.radiusSearch(query, search_rad_sq, matches, params);

        double ztot = 0.0;
        double wtot = 0.0;

        for (const auto& m : matches) {
            const std::size_t j = static_cast<std::size_t>(m.first);
            const double dx = xi - pts.get_x(j);
            const double dy = yi - pts.get_y(j);

            // Square shape: nanoflann gave us the bounding circle; narrow to
            // the actual square footprint.
            if (shape == Shape::Square) {
                if (std::abs(dx) > half_res || std::abs(dy) > half_res) {
                    continue;
                }
            }

            double w = 1.0;
            if (method == SmoothMethod::Gaussian) {
                w = std::exp(-(dx * dx + dy * dy) / twosquaresigma) /
                    twosquaresigmapi;
            }

            ztot += w * pts.get_z(j);
            wtot += w;
        }

        // wtot can only be 0 if `matches` excluded every neighbour (square
        // filter on a degenerate query); fall back to the original z to keep
        // the output finite.
        out[i_u] = (wtot > 0.0) ? (ztot / wtot) : pts.get_z(i_u);
    }

    return out;
}

}  // namespace pylidar::its
