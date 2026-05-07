// PORT NOTE — porting from lidR
// Source:        lidR/src/LAS.cpp:1113-1280  (LAS::segment_trees, the C-side
//                of `li2012`)  +  lidR/src/LAS.cpp:399-480  (the
//                `filter_local_maxima(ws, min_height, circular)` pre-pass that
//                segment_trees calls when R > 0)  +
//                lidR/src/RcppFunction.cpp:91-96  (the `C_li2012` shim).
// lidR commit:   TBD: regen with R env (manual_derivation in M2; will revisit
//                once `tools/regen_fixtures.R` is wired to a real R+lidR env).
// Layout:        N/A — algorithm is point-cloud based, no Matrix2D.
// Indexing:      0-based throughout. lidR mixes 0-based C++ loops with 1-based
//                R-facing tree ids; we keep 1-based **tree ids** (0 = unset)
//                to align with the rest of pylidar (see silva2016 /
//                dalponte2016) and drop NA_INTEGER (no R semantics here).
// NaN:           std::isnan() guard would matter only via input coords; we
//                don't synthesise NaN. -ffast-math intentionally not enabled.
// Threading:     Single-threaded (rd.md §6). lidR's `filter_local_maxima` is
//                `#pragma omp parallel for`; we replicate sequentially because
//                this milestone has no OpenMP.
// Behavior:      1:1 with lidR (rd.md §7).
//   - Distance comparisons are 2-D (xy only); see
//     `lidR/inst/include/lidR/Point.h::sqdistance` and `distance` — both
//     use `sqrt(dx*dx + dy*dy)`. The Z coordinate gates dt1 vs dt2 only.
//   - Disc-inclusion uses lidR's EPSILON slack (`d² ≤ r² + EPSILON`,
//     EPSILON = 1e-8 from lidR/inst/include/lidR/Shapes.h:9). Without it
//     points sitting on the floating-point boundary would be silently
//     excluded from the LMF neighbour set.
//   - `radius` short-circuits the candidate to the "non-tree" set without
//     applying any Li 2012 rule — and importantly, the candidate is **not**
//     added to N_; lidR (LAS.cpp:1216-1219) only flags inN[i] there.
//     Pushing into N_ would pollute later dmin2 lookups and flip the
//     non-LM-rule verdict on borderline points. See fixture
//     li2012_corner_radius_clip.
//   - lidR seeds N with a dummy point at (xmin-100, ymin-100, 0) so dN is
//     always non-empty for `min_element`. We replicate the dummy exactly so
//     edge cases where the cloud diameter exceeds 100 (and dummy distance
//     stops being a "practically infinite" sentinel) match lidR.
//   - The local-maximum pre-pass uses lidR's "first wins" tie-break in
//     equal-z neighbourhoods (sequential row index order in our port; lidR
//     parallelises and is therefore non-deterministic on ties — sequential
//     is a strict refinement, not a divergence).
//   - lidR's "is_uniform cascade NLM" optimisation (`if (!vws && z < Z[i])
//     state[pt.id] = NLM`) is preserved. With a single scalar R, every
//     window has the same radius, and the cascade is provably output-equal
//     to the non-cascading version. Kept verbatim to ease diffing.
//   - When the highest remaining point is below `th_tree`, lidR's outer
//     loop continues but contributes nothing (`inN` is left zero-init →
//     U becomes empty next iteration). We bail with `break`, which is
//     observably identical.

#include "li2012.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace pylidar::core::its {

namespace {

// Match lidR/inst/include/lidR/Shapes.h:9 — slack for floating-point
// boundary inclusion. Used identically in lmf.cpp.
constexpr double EPSILON = 1e-8;

inline double sqdist2d(const PointXYZ& a, const PointXYZ& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Inline replica of lidR/src/LAS.cpp:399 `filter_local_maxima(ws, min_height,
// circular=true)` — uniform window, sequential, with the cascading-NLM
// optimisation. Returns is_lm[i] = true iff pts[i] is a 2-D local max within
// the disc of radius R/2 centered on its own xy. min_height is hardcoded to
// 0 because lidR's pre-pass call site uses 0.
void prepass_local_maxima(const std::vector<PointXYZ>& pts,
                          double R_diameter,
                          std::vector<char>& is_lm) {
    const std::size_t n = pts.size();
    is_lm.assign(n, 0);

    constexpr char UKN = 0;
    constexpr char NLM = 1;
    std::vector<char> state(n, UKN);

    const double hws = R_diameter / 2.0;
    const double r2  = hws * hws;

    for (std::size_t i = 0; i < n; ++i) {
        if (state[i] == NLM) continue;

        const double zi = pts[i].z;
        bool is_max = true;

        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;

            const double dx = pts[j].x - pts[i].x;
            const double dy = pts[j].y - pts[i].y;
            const double d2 = dx * dx + dy * dy;
            // Match lidR Circle::contains (Shapes.h:184) — `d² ≤ r² +
            // EPSILON`, inclusive of the floating-point boundary.
            if (d2 > r2 + EPSILON) continue;

            const double zj = pts[j].z;

            if (zj > zi) {
                is_max = false;
                // No break: lidR keeps iterating to maintain the cascading
                // NLM mark and the equality tie-break.
            }
            // Cascading-NLM (uniform window only — true in this caller).
            if (zj < zi) {
                state[j] = NLM;
            }
            // First-wins equality tie-break.
            if (zj == zi && is_lm[j]) {
                is_max = false;
                break;
            }
        }

        is_lm[i] = is_max ? 1 : 0;
    }
}

}  // namespace

void li2012(const std::vector<PointXYZ>& pts,
            double dt1,
            double dt2,
            double Zu,
            double R,
            double th_tree,
            double radius,
            std::vector<std::int32_t>& ids) {
    const std::size_t ni = pts.size();
    ids.assign(ni, 0);  // 0 = unset.

    if (ni == 0) return;

    const double dt1_sq    = dt1 * dt1;
    const double dt2_sq    = dt2 * dt2;
    const double radius_sq = radius * radius;

    // Local-maximum pre-pass.
    std::vector<char> is_lm;
    if (R > 0.0) {
        prepass_local_maxima(pts, R, is_lm);
    } else {
        is_lm.assign(ni, 1);
    }

    // Z-sorted descending index queue (lidR allocates PointXYZ* heap nodes;
    // we use indices into pts[] — same logic, no heap churn).
    std::vector<std::uint32_t> U;
    U.reserve(ni);
    for (std::uint32_t i = 0; i < ni; ++i) U.push_back(i);
    std::sort(U.begin(), U.end(),
              [&](std::uint32_t a, std::uint32_t b) {
                  return pts[a].z > pts[b].z;
              });

    // lidR's dummy seed for N. Without it, the very first candidate inside
    // the radius gate would have dN.empty() and `*min_element(dN)` would be
    // UB. Coordinates match lidR/LAS.cpp:1166 verbatim.
    double xmin = pts[0].x;
    double ymin = pts[0].y;
    for (const auto& p : pts) {
        if (p.x < xmin) xmin = p.x;
        if (p.y < ymin) ymin = p.y;
    }
    const PointXYZ dummy{xmin - 100.0, ymin - 100.0, 0.0};

    std::int32_t k = 1;  // current tree id (lidR starts at 1)

    while (!U.empty()) {
        const std::uint32_t u_id = U[0];
        const PointXYZ&     u    = pts[u_id];

        // Highest remaining is below th_tree → no further trees will form
        // (z-sorted queue). Equivalent to lidR's no-op iterate-and-empty.
        if (u.z < th_tree) break;

        std::vector<std::uint32_t> P;
        std::vector<std::uint32_t> N_;
        P.reserve(64);
        N_.reserve(64);

        std::vector<char> inN(U.size(), 0);
        P.push_back(u_id);
        ids[u_id] = k;

        const std::size_t nU = U.size();
        for (std::size_t i = 1; i < nU; ++i) {
            const std::uint32_t v_id = U[i];
            const PointXYZ&     v    = pts[v_id];

            const double d_to_u = sqdist2d(v, u);

            if (d_to_u > radius_sq) {
                // lidR/LAS.cpp:1216-1219 only flags inN here — the point is
                // *not* added to N. Pushing it into N_ would pollute later
                // dmin2 lookups and flip non-LM-rule verdicts on borderline
                // points (regression: see fixture li2012_corner_radius_clip).
                inN[i] = 1;
                continue;
            }

            // Distances to current P / N (and to dummy for N).
            double dmin1 = std::numeric_limits<double>::infinity();
            for (std::uint32_t p_id : P) {
                const double d = sqdist2d(v, pts[p_id]);
                if (d < dmin1) dmin1 = d;
            }
            double dmin2 = sqdist2d(v, dummy);  // dummy is always in N
            for (std::uint32_t n_id : N_) {
                const double d = sqdist2d(v, pts[n_id]);
                if (d < dmin2) dmin2 = d;
            }

            const double dt = (v.z > Zu) ? dt2_sq : dt1_sq;

            if (is_lm[v_id]) {
                if (dmin1 > dt || (dmin1 < dt && dmin1 > dmin2)) {
                    inN[i] = 1;
                    N_.push_back(v_id);
                } else {
                    P.push_back(v_id);
                    ids[v_id] = k;
                }
            } else {
                if (dmin1 <= dmin2) {
                    P.push_back(v_id);
                    ids[v_id] = k;
                } else {
                    inN[i] = 1;
                    N_.push_back(v_id);
                }
            }
        }

        // Rebuild U: keep only points marked inN (i.e. rejected from the
        // current tree). lidR drops i==0 (the treetop) by leaving inN[0]
        // false; same here.
        std::vector<std::uint32_t> next_U;
        next_U.reserve(N_.size());
        for (std::size_t i = 0; i < nU; ++i) {
            if (inN[i]) next_U.push_back(U[i]);
        }
        U.swap(next_U);
        ++k;
    }
}

}  // namespace pylidar::core::its
