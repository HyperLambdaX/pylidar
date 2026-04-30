// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::dalponte2016 implementation. See dalponte2016.hpp for
// scope/contract notes.
//
// Algorithm: line-for-line port of C_dalponte2016 (lidR v4.3.2). The
// outer "grown" loop iterates until no pixel changes label in one full
// scan. Each scan:
//   - For every already-labelled inner pixel (skipping the 1-pixel
//     border, same as lidR), look at the 4 cardinal neighbours.
//   - A neighbour joins the crown if it passes five tests:
//       z > th_tree                          (above tree mask)
//       z > h_seed * th_seed                 (vs. seed peak)
//       z > mean_crown_z * th_cr              (vs. running crown mean)
//       z <= h_seed + 0.05 * h_seed          (not above the seed)
//       Chebyshev distance to seed < max_cr   (compactness)
//       region(neighbour) == 0                (not yet labelled)
//   - Writes go into a scratch matrix `region_temp` so within one scan
//     all reads use the previous scan's labels (no order dependence
//     across pixels). At end-of-scan, region_temp is copied back to
//     region. Identical to lidR's std::copy approach.
//
// Two-buffer alternation is intentional and matches lidR. Within one
// scan it is *possible* for two different seeds to both expand into the
// same neighbour — whichever is visited last in row-major order wins,
// because both writes target the same `region_temp(nr, nc)` cell.
// Deterministic by definition (single thread, fixed iteration order).

#include "its/dalponte2016.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace pylidar::its {

namespace {

struct SeedPx {
    int          row;
    int          col;
    std::int32_t id;
};

}  // namespace

common::Matrix2D<std::int32_t> dalponte2016(
    const common::RasterView<double>&    chm,
    const std::vector<common::TreeTop>&  seeds,
    double                               th_seed,
    double                               th_cr,
    double                               th_tree,
    double                               max_cr)
{
    if (chm.data.empty()) {
        throw std::invalid_argument(
            "dalponte2016: chm must have non-zero rows and cols");
    }
    if (!std::isfinite(chm.pixel_size) || chm.pixel_size <= 0.0) {
        throw std::invalid_argument(
            "dalponte2016: chm.pixel_size must be a finite positive number");
    }
    if (!(th_seed >= 0.0 && th_seed <= 1.0)) {
        throw std::invalid_argument(
            "dalponte2016: th_seed must be in [0, 1]");
    }
    if (!(th_cr >= 0.0 && th_cr <= 1.0)) {
        throw std::invalid_argument(
            "dalponte2016: th_cr must be in [0, 1]");
    }
    if (!std::isfinite(th_tree)) {
        throw std::invalid_argument(
            "dalponte2016: th_tree must be finite");
    }
    if (!std::isfinite(max_cr) || max_cr <= 0.0) {
        throw std::invalid_argument(
            "dalponte2016: max_cr must be a finite positive number");
    }

    const std::size_t H = chm.data.rows();
    const std::size_t W = chm.data.cols();
    const int nrow = static_cast<int>(H);
    const int ncol = static_cast<int>(W);

    // NaN→-inf working copy of the canopy. Matches lidR's R-side
    // `Canopy[is.na(Canopy)] <- -Inf` so the `z > th_tree` and other
    // strict-greater tests reject NaN cells without extra branches.
    common::Matrix2D<double> image(H, W);
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    for (std::size_t r = 0; r < H; ++r) {
        for (std::size_t c = 0; c < W; ++c) {
            const double v = chm.data.at(r, c);
            image.at(r, c) = std::isnan(v) ? NEG_INF : v;
        }
    }

    // Rasterise seeds (world XY → pixel) and seed the bookkeeping maps.
    common::Matrix2D<std::int32_t> region(H, W);  // zero-init
    std::unordered_map<std::int32_t, SeedPx> seed_px;
    std::unordered_map<std::int32_t, double> sum_height;
    std::unordered_map<std::int32_t, int>    npixel;

    const double inv_ps = 1.0 / chm.pixel_size;
    for (const auto& s : seeds) {
        if (s.id == 0) continue;  // 0 is the "unlabelled" sentinel
        // col = (x - origin_x) / pixel_size,
        // row = (origin_y - y) / pixel_size  (row 0 is largest y).
        const double col_d = (s.x - chm.origin_x) * inv_ps;
        const double row_d = (chm.origin_y - s.y) * inv_ps;
        if (!std::isfinite(col_d) || !std::isfinite(row_d)) continue;
        const int col_i = static_cast<int>(std::lround(col_d));
        const int row_i = static_cast<int>(std::lround(row_d));
        if (row_i < 0 || row_i >= nrow) continue;
        if (col_i < 0 || col_i >= ncol) continue;

        const std::size_t r_u = static_cast<std::size_t>(row_i);
        const std::size_t c_u = static_cast<std::size_t>(col_i);
        region.at(r_u, c_u) = s.id;
        seed_px[s.id]    = SeedPx{row_i, col_i, s.id};
        sum_height[s.id] = image.at(r_u, c_u);
        npixel[s.id]     = 1;
    }

    // Scratch matrix; starts as a copy of region.
    common::Matrix2D<std::int32_t> region_temp(H, W);
    for (std::size_t r = 0; r < H; ++r) {
        for (std::size_t c = 0; c < W; ++c) {
            region_temp.at(r, c) = region.at(r, c);
        }
    }

    bool grown = true;
    while (grown) {
        grown = false;

        for (int r = 1; r < nrow - 1; ++r) {
            for (int k = 1; k < ncol - 1; ++k) {
                const std::int32_t id =
                    region.at(static_cast<std::size_t>(r),
                              static_cast<std::size_t>(k));
                if (id == 0) continue;

                const auto seed_it = seed_px.find(id);
                if (seed_it == seed_px.end()) continue;  // shouldn't happen
                const SeedPx& seed = seed_it->second;
                const double  h_seed = image.at(
                    static_cast<std::size_t>(seed.row),
                    static_cast<std::size_t>(seed.col));
                const double  mh_crown =
                    sum_height[id] / static_cast<double>(npixel[id]);

                // 4 cardinal neighbours: up, left, right, down. Same
                // ordering as upstream so any "last write wins" tie
                // matches lidR.
                const int neigh[4][2] = {
                    {r - 1, k    },
                    {r,     k - 1},
                    {r,     k + 1},
                    {r + 1, k    },
                };

                for (int n = 0; n < 4; ++n) {
                    const int nr = neigh[n][0];
                    const int nc = neigh[n][1];
                    const std::size_t nr_u = static_cast<std::size_t>(nr);
                    const std::size_t nc_u = static_cast<std::size_t>(nc);
                    const double pz = image.at(nr_u, nc_u);

                    if (!(pz > th_tree)) continue;  // also rejects -inf

                    const bool expand =
                        pz > h_seed * th_seed &&
                        pz > mh_crown * th_cr &&
                        pz <= h_seed + h_seed * 0.05 &&
                        std::abs(seed.row - nr) < max_cr &&
                        std::abs(seed.col - nc) < max_cr &&
                        region.at(nr_u, nc_u) == 0;

                    if (expand) {
                        region_temp.at(nr_u, nc_u) = id;
                        npixel[id]    += 1;
                        sum_height[id] += pz;
                        grown = true;
                    }
                }
            }
        }

        // End-of-scan: promote scratch → working buffer.
        for (std::size_t r = 0; r < H; ++r) {
            for (std::size_t c = 0; c < W; ++c) {
                region.at(r, c) = region_temp.at(r, c);
            }
        }
    }

    return region;
}

}  // namespace pylidar::its
