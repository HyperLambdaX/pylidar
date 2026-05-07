// PORT NOTE — porting from lidR
// Source:        lidR/src/C_dalponte2016.cpp:30-126
// lidR commit:   TBD: regen with R env
// Layout:        lidR uses column-major NumericMatrix; we use row-major
//                Matrix2D<T>. Algorithm is layout-agnostic (4-connected
//                flood fill); translated index-for-index. No transpose.
// Indexing:      0-based throughout. lidR's C++ layer is already 0-based;
//                only the public R-facing wrapper carried 1-based ids
//                (those are dropped here).
// NaN:           std::isnan() is the guard idiom; we do NOT enable
//                -ffast-math (would break NaN comparisons). lidR's R
//                wrapper pre-translates NA → -Inf before invoking the C++
//                routine; we accept either — NaN values fail every "x > t"
//                comparison and are therefore never expanded into.
// Threading:     Single-threaded. No #pragma omp.
// Behavior:      1:1 with lidR (rd.md §7); intentional perf shortcuts deferred.
//                Param renames at our public surface: lidR's th_crown ↔ our
//                th_cr; lidR's DIST ↔ our max_cr.

#include "dalponte2016.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace pylidar::core::its {

namespace {

struct SeedPos {
    std::size_t r;
    std::size_t c;
};

}  // namespace

void dalponte2016(const Matrix2D<double>& chm,
                  const Matrix2D<int32_t>& seeds,
                  Matrix2D<int32_t>& regions,
                  double th_tree,
                  double th_seed,
                  double th_crown,
                  double max_cr) {
    const std::size_t nrow = chm.rows();
    const std::size_t ncol = chm.cols();

    if (seeds.rows() != nrow || seeds.cols() != ncol ||
        regions.rows() != nrow || regions.cols() != ncol) {
        throw std::invalid_argument(
            "dalponte2016: chm, seeds and regions must share shape");
    }

    // Below 3x3 the inner loop range [1, n-1) is empty and no growth is
    // possible. lidR exhibits the same behaviour; we just exit early so
    // unsigned arithmetic on nrow-1/ncol-1 stays safe.
    if (nrow < 3 || ncol < 3) {
        return;
    }

    std::unordered_map<int32_t, SeedPos> seed_pos;
    std::unordered_map<int32_t, double>  sum_height;
    std::unordered_map<int32_t, int32_t> npixel;

    for (std::size_t i = 0; i < nrow; ++i) {
        for (std::size_t j = 0; j < ncol; ++j) {
            const int32_t id = seeds(i, j);
            if (id != 0) {
                seed_pos[id]   = SeedPos{i, j};
                sum_height[id] = chm(i, j);
                npixel[id]     = 1;
            }
        }
    }

    // Two-buffer strategy mirrors lidR's `Region` / `Regiontemp` split:
    // each pass reads from `regions` and writes growth into the temp buffer,
    // then swaps via std::copy at the bottom of the loop.
    std::vector<int32_t> temp_buf(nrow * ncol);
    std::copy(regions.data(), regions.data() + nrow * ncol, temp_buf.begin());
    Matrix2D<int32_t> regiontemp(temp_buf.data(), nrow, ncol);

    bool grown = true;
    while (grown) {
        grown = false;

        for (std::size_t r = 1; r + 1 < nrow; ++r) {
            for (std::size_t k = 1; k + 1 < ncol; ++k) {
                if (regions(r, k) == 0) continue;

                const int32_t  id      = regions(r, k);
                // .at() instead of operator[]: every non-zero label in
                // `regions` was placed by a known seed (registered in the
                // pre-population loop above), so .at() never throws here.
                // Using it defensively means a future refactor that breaks
                // the invariant fails loudly instead of silently inserting
                // a default-constructed entry.
                const SeedPos  seed    = seed_pos.at(id);
                const double   hSeed   = chm(seed.r, seed.c);
                const double   mhCrown = sum_height.at(id) /
                                         static_cast<double>(npixel.at(id));

                const std::size_t nbrs[4][2] = {
                    {r - 1, k},
                    {r,     k - 1},
                    {r,     k + 1},
                    {r + 1, k},
                };

                for (int n = 0; n < 4; ++n) {
                    const std::size_t nr = nbrs[n][0];
                    const std::size_t nc = nbrs[n][1];
                    const double      pz = chm(nr, nc);

                    if (!(pz > th_tree)) continue;

                    // Match lidR's `abs(int) < double`: signed subtraction
                    // is safe here (nrow, ncol fit in long long), and the
                    // promotion to double on the comparison preserves
                    // fractional max_cr values exactly as lidR does.
                    const double dr = static_cast<double>(seed.r) -
                                      static_cast<double>(nr);
                    const double dc = static_cast<double>(seed.c) -
                                      static_cast<double>(nc);

                    const bool expend =
                        pz > hSeed   * th_seed  &&
                        pz > mhCrown * th_crown &&
                        pz <= hSeed + hSeed * 0.05 &&
                        std::fabs(dr) < max_cr   &&
                        std::fabs(dc) < max_cr   &&
                        regions(nr, nc) == 0;

                    if (expend) {
                        regiontemp(nr, nc) = id;
                        npixel.at(id)++;
                        sum_height.at(id) += pz;
                        grown = true;
                    }
                }
            }
        }

        std::copy(temp_buf.begin(), temp_buf.end(), regions.data());
    }
}

}  // namespace pylidar::core::its
