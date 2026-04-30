// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::lmf implementation. See lmf.hpp for scope/contract notes.
//
// Source: lidR v4.3.2 LAS::filter_local_maxima(NumericVector ws, double
// min_height, bool circular) (src/LAS.cpp:399-480). The translation
// deliberately diverges from upstream in two places:
//
//   1. Deterministic tie-break. lidR's inner loop reads `filter[pt.id]`
//      without synchronisation while the outer parallel-for is concurrently
//      writing to it; for two neighbours at exactly the same Z this turns
//      thread-scheduling order into output. We replace the racey check
//      with `z == zi && j < i` — among equal-height points in one window,
//      the lowest-index point wins. Single-threaded behaviour is identical
//      to a serial run; multi-threaded behaviour is now identical to the
//      serial run too.
//
//   2. No cross-iteration `state[j] = NLM` optimisation. lidR fast-paths
//      shorter neighbours of an LM by writing into a shared char-vector
//      from inside the parallel region. With (1) above no longer needed,
//      keeping that write would just be an unsynchronised cross-iteration
//      side effect for no algorithmic value — drop it. Each outer
//      iteration now only writes its own filter[i] slot, so the entire
//      parallel loop is data-race-free.
//
// `zmax` (lidR's variable, never updated after init) is also gone — the
// only thing that lived for was the now-removed shared-state writes.

#include "its/lmf.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common/nanoflann_adaptor.hpp"
#include "common/point_cloud.hpp"

namespace pylidar::its {

namespace {

// Core port of LAS::filter_local_maxima(ws, min_height, circular). Output
// `filter` is sized to pts.n and zero-filled; on return filter[i] != 0
// flags point i as a local maximum. Output is fully deterministic
// regardless of thread count or schedule.
void lmf_filter_impl(
    const common::PointCloudXYZ& pts,
    double                       ws,
    double                       hmin,
    bool                         circular,
    std::vector<char>&           filter)
{
    if (!std::isfinite(ws) || ws <= 0.0) {
        throw std::invalid_argument(
            "lmf: ws must be a finite positive number");
    }
    // hmin is only checked for finiteness (negative is a legitimate
    // "no height filter" choice). Without this guard, hmin = NaN would
    // make every `zi >= hmin` comparison false, every point would be
    // skipped, and the user would see an empty result with no error —
    // the same silent-bypass class Phase 1 fixed for size/sigma. core/
    // is meant to be linkable independent of the Python validator, so
    // the check has to live here.
    if (!std::isfinite(hmin)) {
        throw std::invalid_argument(
            "lmf: hmin must be a finite number");
    }

    const std::size_t n = pts.n;
    filter.assign(n, 0);
    if (n == 0) return;

    const double half_ws = ws / 2.0;
    // Square shape uses the circumscribed circle (half-diagonal) then narrows
    // to the bounding square in the inner loop — same template as
    // smooth_height.cpp.
    const double half_ws_sq    = half_ws * half_ws;
    const double diag_sq       = 2.0 * half_ws_sq;
    const double search_rad_sq = circular ? half_ws_sq : diag_sq;

    common::PointCloudXYZ_KDAdaptor adaptor(pts);
    common::KDTree2D tree(2, adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    const std::ptrdiff_t n_signed = static_cast<std::ptrdiff_t>(n);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n_signed; ++i) {
        const std::size_t i_u = static_cast<std::size_t>(i);
        const double      zi  = pts.get_z(i_u);

        // hmin filter; the `!(zi >= hmin)` form also rejects NaN z
        // (NaN compares unordered, so `NaN >= hmin` is false).
        if (!(zi >= hmin)) continue;

        const double xi = pts.get_x(i_u);
        const double yi = pts.get_y(i_u);
        const double query[2] = {xi, yi};

        std::vector<nanoflann::ResultItem<std::uint32_t, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = false;
        tree.radiusSearch(query, search_rad_sq, matches, params);

        bool is_lm = true;
        for (const auto& m : matches) {
            const std::size_t j = static_cast<std::size_t>(m.first);

            // Square shape: drop neighbours outside the bbox.
            if (!circular) {
                const double dx = pts.get_x(j) - xi;
                const double dy = pts.get_y(j) - yi;
                if (std::abs(dx) > half_ws || std::abs(dy) > half_ws) {
                    continue;
                }
            }

            const double z = pts.get_z(j);

            // Found a strictly higher neighbour: not an LM. Safe to break
            // now — there are no cross-iteration side effects to apply
            // (cf. file header, divergence #2 from lidR).
            if (z > zi) {
                is_lm = false;
                break;
            }

            // Deterministic tie-break: among equal-height points in this
            // window, the lowest-index one wins. j < i_u skips both the
            // self-match (j == i_u) and any neighbour with a higher index.
            if (z == zi && j < i_u) {
                is_lm = false;
                break;
            }
        }

        // Each outer iteration writes its own filter[i_u] only — distinct
        // indices across threads, no race, no critical section needed.
        if (is_lm) filter[i_u] = 1;
    }
}

}  // namespace

std::vector<common::TreeTop> lmf_points(
    const common::PointCloudXYZ& pts,
    double                       ws,
    double                       hmin,
    Shape                        shape)
{
    const bool circular = (shape == Shape::Circular);

    std::vector<char> filter;
    lmf_filter_impl(pts, ws, hmin, circular, filter);

    std::vector<common::TreeTop> tops;
    tops.reserve(filter.size() / 32);  // ITS density is sparse; rough hint.
    for (std::size_t i = 0; i < pts.n; ++i) {
        if (!filter[i]) continue;
        tops.push_back(common::TreeTop{
            pts.get_x(i), pts.get_y(i), pts.get_z(i), /*id=*/0});
    }
    return tops;
}

std::vector<common::TreeTop> lmf_chm(
    const common::RasterView<double>& chm,
    double                            ws,
    double                            hmin,
    Shape                             shape)
{
    if (!std::isfinite(chm.pixel_size) || chm.pixel_size <= 0.0) {
        throw std::invalid_argument(
            "lmf_chm: pixel_size must be a finite positive number");
    }
    if (chm.data.rows() == 0 || chm.data.cols() == 0) {
        throw std::invalid_argument(
            "lmf_chm: chm must have non-zero rows and cols");
    }

    const std::size_t rows = chm.data.rows();
    const std::size_t cols = chm.data.cols();
    const std::size_t cap  = rows * cols;

    // Build a virtual point cloud, one point per non-NaN cell. The XY are
    // absolute world coords; z is the pixel value. lidR's raster_as_las
    // does the same skip-NaN gather (terra::as.points by default drops
    // NA cells).
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    xs.reserve(cap);
    ys.reserve(cap);
    zs.reserve(cap);

    for (std::size_t r = 0; r < rows; ++r) {
        const double y_world = chm.world_y(r);
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = chm.data.at(r, c);
            if (!std::isfinite(v)) continue;  // NaN/Inf cells: skip
            xs.push_back(chm.world_x(c));
            ys.push_back(y_world);
            zs.push_back(v);
        }
    }

    if (xs.empty()) {
        return {};
    }

    common::PointCloudXYZ pc{
        xs.data(), ys.data(), zs.data(),
        xs.size(), /*stride=*/1};

    return lmf_points(pc, ws, hmin, shape);
}

}  // namespace pylidar::its
