// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::silva2016 implementation. See silva2016.hpp for
// scope/contract notes and docs/notes/silva2016-translation-trace.md for
// the line-by-line R→C++ trace (gitignored).
//
// Three passes over the raster:
//
//   Pass 1 (parallel over rows): for each non-NaN cell, query the seed
//     KDTree2D for k=1 nearest neighbour, store (nearest_idx, sqrt_dist)
//     into per-cell scratch buffers. nearest_idx == -1 marks "skip"
//     (NaN cell).
//
//   Pass 2 (serial): scan the raster once, accumulate
//     hmax[g] = max(Z over cells in group g). This step has a per-group
//     reduction and is hard to parallelise without race or per-thread
//     hmax arrays + merge; the workload is a single O(H·W) memory
//     sweep with no math, so serial is the simplest correct choice
//     (decision-3 from the translation trace).
//
//   Pass 3 (parallel over rows): for each non-NaN cell, write the seed
//     id into the result iff the two threshold tests pass.
//
// Notes on the R↔C++ correspondence (cf. silva2016-translation-trace.md):
//
//   * `hmax := max(Z), by = id` is the *Voronoi-cell* max, **not** the
//     seed's own z. We never read seeds[i].z anywhere in the algorithm.
//   * `chmdt[Z >= exclusion*hmax & d <= max_cr_factor*hmax]` uses `>=`
//     and `<=` (inclusive on both sides). dalponte uses strict `>` /
//     `<` — different algorithms, intentionally not "harmonised".
//   * lidR pre-crops treetops to the chm bbox via sf::st_crop. We do
//     the equivalent up front: any seed whose world XY is outside the
//     chm bbox (including the 0.5-pixel half-skirt around the pixel
//     centres) is dropped before KDTree construction.

#include "its/silva2016.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/nanoflann_adaptor.hpp"
#include "common/point_cloud.hpp"

namespace pylidar::its {

common::Matrix2D<std::int32_t> silva2016(
    const common::RasterView<double>&    chm,
    const std::vector<common::TreeTop>&  seeds,
    double                               max_cr_factor,
    double                               exclusion)
{
    if (chm.data.empty()) {
        throw std::invalid_argument(
            "silva2016: chm must have non-zero rows and cols");
    }
    if (!std::isfinite(chm.pixel_size) || chm.pixel_size <= 0.0) {
        throw std::invalid_argument(
            "silva2016: chm.pixel_size must be a finite positive number");
    }
    if (!std::isfinite(max_cr_factor) || max_cr_factor <= 0.0) {
        throw std::invalid_argument(
            "silva2016: max_cr_factor must be a finite positive number");
    }
    // Open interval (0, 1) — matches lidR's assert_all_are_in_open_range.
    // Both endpoints are *errors* (exclusion=0 would make every cell
    // pass the height threshold; exclusion=1 would require Z = hmax
    // exactly).
    if (!std::isfinite(exclusion) || exclusion <= 0.0 || exclusion >= 1.0) {
        throw std::invalid_argument(
            "silva2016: exclusion must lie in the open interval (0, 1)");
    }

    const std::size_t H = chm.data.rows();
    const std::size_t W = chm.data.cols();

    // 1. Pre-crop seeds to chm bbox + finiteness check (lidR
    //    crop_special_its + the core's own self-check on raw inputs).
    //    The bbox covers the *full pixel footprint*, so a seed sitting
    //    on a corner pixel's outer edge still qualifies — same as
    //    sf::st_crop on a stars/SpatRaster bbox.
    const double half_ps = 0.5 * chm.pixel_size;
    const double xmin    = chm.origin_x - half_ps;
    const double xmax    = chm.origin_x + (static_cast<double>(W) - 0.5)
                                          * chm.pixel_size;
    const double ymin    = chm.origin_y - (static_cast<double>(H) - 0.5)
                                          * chm.pixel_size;
    const double ymax    = chm.origin_y + half_ps;

    std::vector<common::TreeTop> valid_seeds;
    valid_seeds.reserve(seeds.size());
    for (const auto& s : seeds) {
        if (s.id == 0) continue;
        // Non-finite XY: throw (matches hpp contract + Phase 1/2 template +
        // spec §6.4). Silent drop here would produce mysterious empty
        // results when core/ is linked outside the Python wrapper. The
        // public Python API already rejects NaN/Inf XY at
        // _validate.ensure_seeds_xyzid, so user-facing paths see this
        // throw only when they bypass the wrapper (e.g., direct _core call).
        if (!std::isfinite(s.x) || !std::isfinite(s.y)) {
            throw std::invalid_argument(
                "silva2016: seed XY must be finite (NaN/Inf rejected; "
                "use pylidar.segment_silva2016 or call "
                "_validate.ensure_seeds_xyzid up front)");
        }
        if (s.x < xmin || s.x > xmax) continue;
        if (s.y < ymin || s.y > ymax) continue;
        valid_seeds.push_back(s);
    }

    common::Matrix2D<std::int32_t> result(H, W);  // zero-initialised
    if (valid_seeds.empty()) {
        return result;
    }

    // 2. Build the seed point cloud + KDTree2D. Three contiguous double
    //    vectors so PointCloudXYZ uses stride=1; z is unused but the
    //    POD struct still needs a non-null z pointer.
    const std::size_t M = valid_seeds.size();
    std::vector<double> seed_xs(M);
    std::vector<double> seed_ys(M);
    std::vector<double> seed_zs(M);
    for (std::size_t i = 0; i < M; ++i) {
        seed_xs[i] = valid_seeds[i].x;
        seed_ys[i] = valid_seeds[i].y;
        seed_zs[i] = valid_seeds[i].z;
    }
    common::PointCloudXYZ seed_pc{
        seed_xs.data(), seed_ys.data(), seed_zs.data(),
        M, /*stride=*/1};
    common::PointCloudXYZ_KDAdaptor adaptor(seed_pc);
    common::KDTree2D tree(2, adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    // 3. Pass 1 — parallel KNN. nearest_idx and cell_dist are
    //    H·W double/int32 scratch buffers (column-major Matrix2D);
    //    nearest_idx == -1 marks NaN cells so passes 2 and 3 can
    //    skip them in O(1).
    common::Matrix2D<std::int32_t> nearest_idx(H, W);
    common::Matrix2D<double>       cell_dist  (H, W);

    // Initialise nearest_idx to -1 in one shot. Matrix2D zero-inits int32
    // by default (we want -1 for "no seed yet / NaN cell").
    for (std::size_t r = 0; r < H; ++r) {
        for (std::size_t c = 0; c < W; ++c) {
            nearest_idx.at(r, c) = -1;
        }
    }

    const std::ptrdiff_t H_signed = static_cast<std::ptrdiff_t>(H);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t r_s = 0; r_s < H_signed; ++r_s) {
        const std::size_t r = static_cast<std::size_t>(r_s);
        const double y_world = chm.world_y(r);
        for (std::size_t c = 0; c < W; ++c) {
            const double v = chm.data.at(r, c);
            if (!std::isfinite(v)) continue;  // NaN/Inf cell: skip
            const double x_world = chm.world_x(c);
            const double query[2] = {x_world, y_world};

            std::uint32_t idx = 0;
            double        sq_d = 0.0;
            nanoflann::KNNResultSet<double, std::uint32_t> rs(1);
            rs.init(&idx, &sq_d);
            tree.findNeighbors(rs, query, nanoflann::SearchParameters{});

            nearest_idx.at(r, c) = static_cast<std::int32_t>(idx);
            cell_dist  .at(r, c) = std::sqrt(sq_d);
        }
    }

    // 4. Pass 2 — serial hmax accumulation per Voronoi group.
    //    `chmdt[, hmax := max(Z), by = id]` in R, where the group is
    //    "cells whose nearest seed is i". hmax stays at -inf for any
    //    seed with no cells (all-NaN region or seed alone on an
    //    isolated island); pass 3 then leaves all of that seed's
    //    crown empty since `Z >= exclusion * (-inf)` is true but
    //    the `dist <= max_cr_factor * (-inf)` test would need
    //    dist <= -inf which is false everywhere.
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> hmax(M, NEG_INF);
    for (std::size_t r = 0; r < H; ++r) {
        for (std::size_t c = 0; c < W; ++c) {
            const std::int32_t i = nearest_idx.at(r, c);
            if (i < 0) continue;
            const double v = chm.data.at(r, c);
            if (v > hmax[static_cast<std::size_t>(i)]) {
                hmax[static_cast<std::size_t>(i)] = v;
            }
        }
    }

    // 5. Pass 3 — parallel threshold + write. Each cell only reads
    //    hmax[i] (read-only after pass 2) and writes its own
    //    result.at(r, c) (distinct indices across threads), so no
    //    sync is needed.
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t r_s = 0; r_s < H_signed; ++r_s) {
        const std::size_t r = static_cast<std::size_t>(r_s);
        for (std::size_t c = 0; c < W; ++c) {
            const std::int32_t i = nearest_idx.at(r, c);
            if (i < 0) continue;  // NaN cell — leave 0
            const double v = chm.data.at(r, c);
            const double d = cell_dist.at(r, c);
            const double h = hmax[static_cast<std::size_t>(i)];
            // R: Z >= exclusion*hmax & d <= max_cr_factor*hmax
            // (inclusive on both sides — different from dalponte).
            if (v >= exclusion * h && d <= max_cr_factor * h) {
                result.at(r, c) = valid_seeds[
                    static_cast<std::size_t>(i)].id;
            }
        }
    }

    return result;
}

}  // namespace pylidar::its
