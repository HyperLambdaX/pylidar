// PORT NOTE — porting from EBImage
// Source:        EBImage src/watershed.cpp (206 lines, full file) and
//                EBImage src/tools.h (PointXY / POINT_FROM_INDEX /
//                DISTANCE_XY / INDEX_FROM_XY macros).
// Pinned to commit SHAs (devel branch, looked up 2026-05-13):
//   src/watershed.cpp @ eb528bda72038be848cdc681c69bd16dedad4d32 (2024-10-14)
//     "Apply patch from Kurt Hornik"
//     https://github.com/aoles/EBImage/blob/eb528bda72038be848cdc681c69bd16dedad4d32/src/watershed.cpp
//   src/tools.h       @ 635e8128c2ed29160f580479a300e59819a17144 (2017-04-12)
//     https://github.com/aoles/EBImage/blob/635e8128c2ed29160f580479a300e59819a17144/src/tools.h
// R-side validation:  byte-exact comparison against EBImage::watershed in R is
//                     pending (no R env on the build host). When R becomes
//                     available, drop a few `tests/fixtures/watershed_ext_R_baseline_*.npz`
//                     and compare with `np.array_equal` — the kernel itself
//                     should not need changes.
// lidR caller:   lidR R/algorithm-its.R:328-377 (`watershed` ITS plugin):
//                  Canopy[Canopy < th_tree | is.na] <- 0
//                  Crowns <- EBImage::watershed(Canopy, tol, ext)
//                That R wrapper does the NaN/threshold cleaning *before*
//                calling EBImage; we keep the same contract: this kernel
//                expects a NaN-free CHM with background already zeroed.
// Layout:        EBImage column-major (nx=nrow, index = x + y*nx) → ours
//                row-major (W cols, index = r*W + c). The box neighbourhood
//                is symmetric in both layouts; the only behavioural impact
//                is plateau scan order — see "Sort" below.
// Sort:          EBImage uses R's `rsort_with_index` (non-stable). We use
//                std::stable_sort with row-major index as a deterministic
//                tie-breaker. On no-tie inputs this matches EBImage's
//                deterministic case exactly; on plateaus our output is
//                deterministic and reproducible while EBImage's is not.
// Indexing:      0-based throughout (lidR's C layer is 0-based; only its R
//                surface carries 1-based ids, which the EBImage src does not
//                use).
// NaN:           Trust the upstream R-side cleaning. NaN inputs have
//                undefined behaviour. lidR's wrapper pre-zeros them.
// Threading:     Single-threaded. EBImage's only OMP candidate is the outer
//                `nz` (frame) loop, which we don't carry.

#include "watershed_ext.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <stdexcept>
#include <vector>

namespace pylidar::core::its {

namespace {

struct TheSeed {
    std::size_t index;
    int32_t     seed;
};

// EBImage's check_multiple (src/watershed.cpp:151-198), translated index-for-index.
// `frame` carries the working seed-id raster (doubles holding ints, mirroring
// EBImage's dual use of the same buffer); `src` is the immutable height array.
//
// Single-neighbour case: trivially return the only neighbour. Multi-neighbour:
// (1) find the steepest drop (maxdiff) — that's the default winner; (2) among
// neighbours with drop ≥ tolerance, override with the closest. (3) Merge every
// other neighbour with drop < tolerance into the winner: relabel its cells in
// `frame` and erase its seed record. EBImage uses a list scan to find a seed
// by id; we do the same to preserve the exact merge ordering, even though a
// hashmap would be O(1) per lookup.
int32_t check_multiple(std::vector<double>&       frame,
                       const double*              src,
                       std::size_t                ind,
                       const std::list<int32_t>&  nb,
                       std::list<TheSeed>&        seeds,
                       double                     tolerance,
                       std::size_t                H,
                       std::size_t                W) {
    if (nb.size() == 1) return nb.front();
    if (nb.size() <  1) return 0;  // EBImage "dumb protection".

    auto find_seed = [&](int32_t v) -> std::list<TheSeed>::iterator {
        for (auto it = seeds.begin(); it != seeds.end(); ++it) {
            if (it->seed == v) return it;
        }
        return seeds.end();
    };

    const std::size_t pt_r = ind / W;
    const std::size_t pt_c = ind % W;

    // EBImage uses `long double` for the distance accumulator; mirror it so
    // ties at near-equal distances resolve identically.
    long double dist    = std::numeric_limits<long double>::max();
    double      maxdiff = 0.0;
    int32_t     res     = 0;

    for (auto it = nb.begin(); it != nb.end(); ++it) {
        auto sit = find_seed(*it);
        if (sit == seeds.end()) continue;
        const double diff = std::fabs(src[ind] - src[sit->index]);
        if (diff > maxdiff) {
            maxdiff = diff;
            // EBImage: assign res to the steepest UNTIL/UNLESS later overridden
            // by a closer-than-tolerance neighbour.
            if (dist == std::numeric_limits<long double>::max()) {
                res = *it;
            }
        }
        if (diff >= tolerance) {
            const std::size_t sr = sit->index / W;
            const std::size_t sc = sit->index % W;
            const long double dr = static_cast<long double>(pt_r) - static_cast<long double>(sr);
            const long double dc = static_cast<long double>(pt_c) - static_cast<long double>(sc);
            const long double distx = std::sqrt(dr * dr + dc * dc);
            if (distx < dist) {
                dist = distx;
                res  = *it;
            }
        }
    }

    // Merge the rest into res: relabel frame cells + erase the seed record.
    // Iterating the original `nb` list: erasures touch `seeds`, not `nb`, so
    // iterators stay valid.
    for (auto it = nb.begin(); it != nb.end(); ++it) {
        if (*it == res) continue;
        auto sit = find_seed(*it);
        if (sit == seeds.end()) continue;
        if (std::fabs(src[ind] - src[sit->index]) >= tolerance) continue;
        const int32_t target_seed = *it;
        const std::size_t N = H * W;
        for (std::size_t pi = 0; pi < N; ++pi) {
            if (static_cast<int32_t>(frame[pi]) == target_seed) {
                frame[pi] = static_cast<double>(res);
            }
        }
        seeds.erase(sit);
    }
    return res;
}

}  // namespace

void watershed_ext(const Matrix2D<double>& chm,
                   double                  tolerance,
                   int                     ext,
                   Matrix2D<int32_t>&      out) {
    if (chm.rows() != out.rows() || chm.cols() != out.cols()) {
        throw std::invalid_argument(
            "watershed_ext: chm and out must share shape");
    }
    if (!(tolerance >= 0.0)) {
        throw std::invalid_argument("watershed_ext: tolerance must be >= 0");
    }
    if (ext < 1) {
        throw std::invalid_argument("watershed_ext: ext must be >= 1");
    }

    const std::size_t H = chm.rows();
    const std::size_t W = chm.cols();
    const std::size_t N = H * W;

    // Initialise output to 0 (background sentinel).
    int32_t* tgt = out.data();
    for (std::size_t i = 0; i < N; ++i) tgt[i] = 0;

    if (N == 0) return;

    const double* src = chm.data();

    // ── Build sort index (descending by height). ──
    // EBImage: `frame[i] = -src[i]; rsort_with_index(frame, index, N)` — sort
    // ascending negated values = descending originals. We use std::stable_sort
    // by negated key to be deterministic on plateaus (R's rsort_with_index
    // is not stable; see PORT NOTE).
    std::vector<std::size_t> index(N);
    for (std::size_t i = 0; i < N; ++i) index[i] = i;
    std::stable_sort(index.begin(), index.end(),
        [&](std::size_t a, std::size_t b) {
            // Sort by descending src; tiebreak by ascending row-major index
            // (stable_sort already preserves original order, but be explicit).
            return src[a] > src[b];
        });

    // ── Working frame: doubles, dual-purpose. ──
    // Negative values mark unassigned cells (EBImage stores `-src` then uses
    // (int)x < 1 to detect unassigned); positive integer values are seed ids
    // (cast to int via truncation). A value of 0 marks background — never
    // becomes a label (the main loop only enters when src > 0).
    // EBImage's `(src[i]==0 ? 0 : -src[i])` trick avoids a -0.0 that would
    // not compare equal to +0.0 with bitwise checks; we replicate it.
    std::vector<double> frame(N);
    for (std::size_t i = 0; i < N; ++i) {
        frame[i] = (src[i] == 0.0) ? 0.0 : -src[i];
    }

    std::list<TheSeed>      seeds;
    std::list<std::size_t>  equals;
    std::list<int32_t>      nb;
    int32_t                 topseed = 0;

    // ── Outer loop: scan sorted index from highest to lowest. ──
    // EBImage: `for (i = 0; i < N && src[index[i]] > BG;)` — BG = 0.
    std::size_t i = 0;
    while (i < N && src[index[i]] > 0.0) {
        // Pool all pixels at this exact source value into `equals`.
        const std::size_t ind0 = index[i];
        equals.push_back(ind0);
        ++i;
        while (i < N && src[index[i]] == src[ind0]) {
            equals.push_back(index[i]);
            ++i;
        }

        while (!equals.empty()) {
            // Inner pass: try to assign every queue member by neighbourhood
            // lookup. EBImage iterates with `for (j = 0; j < equals.size(); )`,
            // resetting j to 0 on every successful assignment so re-tries can
            // benefit from newly-assigned neighbours. We mirror exactly.
            std::size_t j = 0;
            std::size_t qsize = equals.size();
            while (j < qsize) {
                const std::size_t cur = equals.front();
                equals.pop_front();
                nb.clear();

                const std::size_t r = cur / W;
                const std::size_t c = cur % W;
                for (int dr = -ext; dr <= ext; ++dr) {
                    for (int dc = -ext; dc <= ext; ++dc) {
                        if (dr == 0 && dc == 0) continue;
                        const long long rr = static_cast<long long>(r) + dr;
                        const long long cc = static_cast<long long>(c) + dc;
                        if (rr < 0 || cc < 0 ||
                            rr >= static_cast<long long>(H) ||
                            cc >= static_cast<long long>(W)) {
                            continue;
                        }
                        const std::size_t indxy = static_cast<std::size_t>(rr) * W
                                                + static_cast<std::size_t>(cc);
                        const int32_t nbseed = static_cast<int32_t>(frame[indxy]);
                        if (nbseed < 1) continue;
                        // Push unique nbseeds only.
                        bool isin = false;
                        for (auto v : nb) {
                            if (v == nbseed) { isin = true; break; }
                        }
                        if (!isin) nb.push_back(nbseed);
                    }
                }

                if (nb.empty()) {
                    // No assigned neighbour yet — re-queue and advance j.
                    equals.push_back(cur);
                    ++j;
                    continue;
                }
                // Assign this pixel; merging may relabel other frame cells.
                frame[cur] = static_cast<double>(
                    check_multiple(frame, src, cur, nb, seeds, tolerance, H, W));
                // EBImage resets j → 0 after every assignment to retry stragglers.
                j     = 0;
                qsize = equals.size();
            }
            // Inner pass done with no assignment possible → promote head to a
            // new seed and restart the assignment loop.
            if (!equals.empty()) {
                ++topseed;
                const std::size_t newind = equals.front();
                equals.pop_front();
                frame[newind] = static_cast<double>(topseed);
                seeds.push_back({newind, topseed});
            }
        }
    }

    // ── Compaction: surviving seeds → contiguous 1..K labels. ──
    // EBImage walks `seeds` in insertion order, assigning new labels 1..K and
    // building a finseed[old_id-1] = new_id lookup. Same here.
    if (topseed == 0) return;
    std::vector<int32_t> finseed(static_cast<std::size_t>(topseed), 0);
    int32_t k = 0;
    for (const auto& s : seeds) {
        finseed[static_cast<std::size_t>(s.seed) - 1] = ++k;
    }
    for (std::size_t pi = 0; pi < N; ++pi) {
        const int32_t v = static_cast<int32_t>(frame[pi]);
        tgt[pi] = (v > 0 && v <= topseed) ? finseed[static_cast<std::size_t>(v) - 1] : 0;
    }
}

}  // namespace pylidar::core::its
