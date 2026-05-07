// PORT NOTE — porting from lidR
// Source:        lidR/src/LAS.cpp:112-179 (`z_smooth(size, method, shape,
//                sigma)`) + lidR/src/RcppFunction.cpp:44-50 (the `C_smooth`
//                shim) + lidR/R/smooth_height.R:34-56 (R wrapper which
//                defaults `sigma = size/6` and maps method/shape strings
//                to the small ints the C function takes).
// lidR commit:   TBD: regen with R env (manual_derivation in M3).
// Layout:        Operates on a flat point cloud (xyz). No raster involved.
//                lidR's `Z` is a NumericVector and we mirror that with a
//                std::vector<double>. No transpose / row-vs-col concerns.
// Indexing:      0-based throughout.
// NaN:           lidR does no NaN-guarding (LAS.cpp:148-167); a NaN
//                neighbour z poisons the running sum. We match: caller is
//                expected to filter NaNs upstream. Uses std::isnan() only
//                where lidR does (it does not, here). -ffast-math
//                intentionally not enabled (spec §7).
// Threading:     Sequential. lidR uses `#pragma omp parallel for
//                num_threads(ncpu)` and a `#pragma omp critical` for the
//                Z_out write-back. Sequential output is a deterministic
//                refinement of the parallel trace — every point's smoothed
//                value is independent of write order, so `omp critical` is
//                purely about the parallel write-back, not algorithmic
//                semantics. Spec §M2 / rd.md §6 explicitly accept dropping
//                OpenMP for this port.
// Behavior:      1:1 with lidR (rd.md §7).
//   - Window inclusion: lidR builds `Rectangle(x±half_res, y±half_res)` or
//     `Circle(x, y, half_res)` and queries SpatialIndex with `tree.lookup`.
//     `Rectangle::contains` and `Circle::contains` (Shapes.h:80,184) both
//     bake in EPSILON = 1e-8 slack — included here.
//   - Self-inclusion: `tree.lookup` includes the query point i itself
//     (lidR has no `j == i` guard — see LAS.cpp:152-167). We include i too.
//   - Average vs Gaussian: identical formulas to LAS.cpp:154-163.
//   - lidR R wrapper has a known typo at smooth_height.R:45 (`if (method
//     == "circle")` instead of `if (shape == "circle")`) which makes the
//     R-side wrapper always fall through to `shape <- 2` (i.e. always
//     Circle in C). Our pylidar entry exposes shape correctly because we
//     translate LAS.cpp directly, not the R wrapper. Documented in
//     findings.md; not "fixing the upstream bug" per spec §9 — we just
//     don't import the buggy R-side mapping in the first place.

#include "chm_smooth.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "kdtree.hpp"

namespace pylidar::core::its {

namespace {

// Match lidR/inst/include/lidR/Shapes.h:9 — slack for floating-point
// boundary inclusion. Mirrors the EPSILON used in lmf.cpp / li2012.cpp.
constexpr double EPSILON = 1e-8;

}  // namespace

void chm_smooth(const std::vector<PointXYZ>& pts,
                double size,
                SmoothMethod method,
                SmoothShape  shape,
                double sigma,
                std::vector<double>& z_out) {
    const std::size_t n = pts.size();
    z_out.assign(n, 0.0);
    if (n == 0) return;

    const double half_res = size / 2.0;
    const double twosigsq    = 2.0 * sigma * sigma;
    const double twosigsq_pi = twosigsq * M_PI;

    // Build a 2D KD-tree over xy.
    std::vector<double> xy(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        xy[i * 2 + 0] = pts[i].x;
        xy[i * 2 + 1] = pts[i].y;
    }
    KdTree2D tree(xy.data(), n);

    std::vector<KdTree2D::IndexType> hits;

    for (std::size_t i = 0; i < n; ++i) {
        // Square shape: query bounding circle (radius half_res*√2), then
        // post-filter to box. Circle: query disc widened to absorb the
        // EPSILON slack of lidR's `d² ≤ r² + EPSILON` test.
        const double query_r2 =
            (shape == SmoothShape::Square) ? 2.0 * half_res * half_res + EPSILON
                                           : half_res * half_res + EPSILON;
        const double query_r  = std::sqrt(query_r2);
        tree.radius_search(pts[i].x, pts[i].y, query_r, hits);

        double ztot = 0.0;
        double wtot = 0.0;

        for (KdTree2D::IndexType j_u : hits) {
            const std::size_t j = static_cast<std::size_t>(j_u);
            const double dx = pts[j].x - pts[i].x;
            const double dy = pts[j].y - pts[i].y;

            // Square shape: drop hits outside the box (hits sit in the
            // bounding circle, may exceed |dx|, |dy| ≤ half_res + EPSILON).
            // Matches lidR/Shapes.h:80-86 Rectangle::contains.
            if (shape == SmoothShape::Square) {
                if (std::fabs(dx) > half_res + EPSILON ||
                    std::fabs(dy) > half_res + EPSILON) continue;
            }

            double w;
            if (method == SmoothMethod::Average) {
                w = 1.0;
            } else {
                const double d2 = dx * dx + dy * dy;
                w = (1.0 / twosigsq_pi) * std::exp(-d2 / twosigsq);
            }

            ztot += w * pts[j].z;
            wtot += w;
        }

        // lidR has no wtot==0 guard — relies on the query point itself
        // always being a hit (self-inclusion via SpatialIndex). We match.
        z_out[i] = ztot / wtot;
    }
}

}  // namespace pylidar::core::its
