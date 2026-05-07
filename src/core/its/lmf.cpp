// PORT NOTE — porting from lidR
// Source:        lidR/src/LAS.cpp:399-480  (`filter_local_maxima(ws,
//                min_height, circular)`, the C-side of `lmf()` for point
//                clouds)  +  lidR/src/RcppFunction.cpp:36-42  (the `C_lmf`
//                shim)  +  lidR/R/algorithm-itd.R:72-107  (the R-side `lmf`
//                wrapper that materialises the per-point ws array from a
//                callable). For the CHM path lidR has no dedicated C
//                routine; `locate_trees(SpatRaster, lmf(...))` calls
//                `raster_as_las(chm)` (each non-NA cell becomes a point at
//                cell-centre coords) and reuses `C_lmf`. We implement the
//                CHM path as a direct raster scan to avoid the materialised
//                point cloud — output is identical because the cell-centre
//                grid is regular and our window inclusion test mirrors
//                lidR's.
// lidR commit:   TBD: regen with R env (manual_derivation in M2).
// Layout:        Row-major Matrix2D<double> for the CHM. lidR doesn't carry
//                a Matrix2D for this code path — it works on flattened
//                cell-centre points. Loops are translated index-for-index;
//                no transpose required (rd.md §3 / spec §7).
// Indexing:      0-based throughout. Output (rows, cols) are 0-based pixel
//                indices.
// NaN:           Cells with NaN in CHM are treated as nodata: never an LM,
//                never a neighbour. Uses std::isnan(); -ffast-math
//                intentionally not enabled.
// Threading:     Single-threaded. lidR uses `#pragma omp parallel for`; we
//                do not (rd.md §6).
// Behavior:      1:1 with lidR (rd.md §7).
//   - Window definition: lidR uses `Rectangle::contains` (`p >= xmin -
//     EPSILON && p <= xmax + EPSILON` etc.) for square and
//     `Circle::contains` (`d² ≤ r² + EPSILON`) for circular, where
//     EPSILON = 1e-8 (lidR/inst/include/lidR/Shapes.h:9). We match the
//     EPSILON slack so floating-point boundary points are inclusive on
//     both sides, consistent with lidR.
//   - hmin pre-pass: lidR sets `state[i] = NLM` if Z[i] < hmin (line 413).
//     Replicated.
//   - vws cascading-NLM: lidR turns the cascade off when ws is per-point
//     (line 457: `if (!vws && z < Z[i]) state[pt.id] = NLM`). The bindings
//     layer hands us `is_uniform` so we can do the same.
//   - Equality tie-break: lidR's `if (z == zmax && filter[pt.id]) ...
//     break` is sequence-order dependent. lidR runs in parallel, so the
//     tie-break is effectively non-deterministic; we run sequentially in
//     point-index order, which is one valid serialisation of the parallel
//     trace and is reproducible. Spec §M2 + rd.md §7 explicitly accept
//     deterministic refinements of non-deterministic upstream behaviour.

#include "lmf.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kdtree.hpp"

namespace pylidar::core::its {

namespace {

constexpr char UKN = 0;
constexpr char NLM = 1;

// Match lidR/inst/include/lidR/Shapes.h:9 — slack for floating-point
// boundary inclusion. Mirrors the EPSILON used in li2012.cpp.
constexpr double EPSILON = 1e-8;

}  // namespace

void lmf_points(const std::vector<PointXYZ>& pts,
                const std::vector<double>&  ws,
                double hmin,
                LmfShape shape,
                bool is_uniform,
                std::vector<char>& lm) {
    const std::size_t n = pts.size();
    lm.assign(n, 0);

    if (n == 0) return;

    // Pre-mark NLM for points below hmin (lidR/LAS.cpp:413).
    std::vector<char> state(n, UKN);
    for (std::size_t i = 0; i < n; ++i) {
        if (pts[i].z < hmin) state[i] = NLM;
    }

    // Build a 2D KD-tree over xy. We need a contiguous (n, 2) buffer; pts is
    // an array of PointXYZ structs, so flatten once (small one-time cost).
    std::vector<double> xy(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        xy[i * 2 + 0] = pts[i].x;
        xy[i * 2 + 1] = pts[i].y;
    }
    KdTree2D tree(xy.data(), n);

    // For square shape we radius-search the bounding circle (radius =
    // hws*sqrt(2)) and post-filter to the box. For circular we expand the
    // disc radius by sqrt(EPSILON / hws²) to absorb the EPSILON slack
    // (kdtree's `d² ≤ r²` is strict — we widen r so the inner test catches
    // any point with `d² ≤ hws² + EPSILON`).
    std::vector<KdTree2D::IndexType> hits;

    for (std::size_t i = 0; i < n; ++i) {
        if (state[i] == NLM) continue;

        const double zi  = pts[i].z;
        const double hws = ws[i] / 2.0;

        const double query_r2 =
            (shape == LmfShape::Square) ? 2.0 * hws * hws + EPSILON
                                        : hws * hws + EPSILON;
        const double query_r  = std::sqrt(query_r2);
        tree.radius_search(pts[i].x, pts[i].y, query_r, hits);

        bool is_max = true;

        for (KdTree2D::IndexType j_u : hits) {
            const std::size_t j = static_cast<std::size_t>(j_u);
            if (j == i) continue;

            // Square shape: drop hits outside the box (hits are inside the
            // bounding circle, but may exceed |dx|, |dy| ≤ hws + EPSILON).
            // EPSILON matches lidR/Shapes.h:84-85 Rectangle::contains.
            if (shape == LmfShape::Square) {
                const double dx = std::fabs(pts[j].x - pts[i].x);
                const double dy = std::fabs(pts[j].y - pts[i].y);
                if (dx > hws + EPSILON || dy > hws + EPSILON) continue;
            }

            const double zj = pts[j].z;

            if (zj > zi) {
                is_max = false;
                // No break — we still need to apply cascading NLM and the
                // equality tie-break (lidR/LAS.cpp:450 onwards).
            }
            // Cascading NLM: only when ws is uniform across all points.
            if (is_uniform && zj < zi) {
                state[j] = NLM;
            }
            // First-wins equality tie-break.
            if (zj == zi && lm[j]) {
                is_max = false;
                break;
            }
        }

        lm[i] = is_max ? 1 : 0;
    }
}

void lmf_chm(const Matrix2D<double>& chm,
             double ws,
             double hmin,
             LmfShape shape,
             std::vector<std::int32_t>& rows,
             std::vector<std::int32_t>& cols) {
    rows.clear();
    cols.clear();

    const std::size_t H = chm.rows();
    const std::size_t W = chm.cols();
    if (H == 0 || W == 0) return;

    const double hws  = ws / 2.0;
    // Window half-extent in pixels (covers the disc / box). Inclusive
    // upper bound matches lidR's Rectangle/Circle ≤ semantics.
    const std::ptrdiff_t hws_int = static_cast<std::ptrdiff_t>(std::floor(hws));
    const double hws2 = hws * hws;

    std::vector<char> state(H * W, UKN);
    std::vector<char> lm_grid(H * W, 0);

    auto idx = [W](std::size_t r, std::size_t c) { return r * W + c; };

    // Pre-mark NLM (z < hmin or NaN).
    for (std::size_t r = 0; r < H; ++r) {
        for (std::size_t c = 0; c < W; ++c) {
            const double v = chm(r, c);
            if (std::isnan(v) || v < hmin) state[idx(r, c)] = NLM;
        }
    }

    // Per-cell sweep extracted as a lambda so the equality tie-break can
    // `return` cleanly out of both nested neighbour loops (mirroring lidR's
    // single-level `break`).
    auto is_local_max = [&](std::size_t r, std::size_t c) -> bool {
        const double zi = chm(r, c);

        const std::ptrdiff_t r_lo =
            std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(r) - hws_int);
        const std::ptrdiff_t r_hi =
            std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(H) - 1,
                                     static_cast<std::ptrdiff_t>(r) + hws_int);
        const std::ptrdiff_t c_lo =
            std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(c) - hws_int);
        const std::ptrdiff_t c_hi =
            std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(W) - 1,
                                     static_cast<std::ptrdiff_t>(c) + hws_int);

        bool is_max = true;
        for (std::ptrdiff_t rr = r_lo; rr <= r_hi; ++rr) {
            for (std::ptrdiff_t cc = c_lo; cc <= c_hi; ++cc) {
                if (rr == static_cast<std::ptrdiff_t>(r) &&
                    cc == static_cast<std::ptrdiff_t>(c)) continue;

                const double dr = static_cast<double>(rr) - static_cast<double>(r);
                const double dc = static_cast<double>(cc) - static_cast<double>(c);

                if (shape == LmfShape::Circular) {
                    // Match lidR Circle::contains (Shapes.h:184): `d² ≤ r²
                    // + EPSILON` — inclusive of float-boundary cells.
                    if (dr * dr + dc * dc > hws2 + EPSILON) continue;
                }
                // Square: the outer integer box already enforces |dr|,|dc| ≤
                // hws_int = floor(hws). For integer-offset CHM cells the
                // EPSILON slack from lidR/Shapes.h:80-86 Rectangle::contains
                // would never include an extra cell anyway (next integer
                // step is +1, much larger than EPSILON), so we skip the
                // explicit `+ EPSILON` test here — output identical.

                const std::size_t j = idx(static_cast<std::size_t>(rr),
                                           static_cast<std::size_t>(cc));
                const double zj = chm(static_cast<std::size_t>(rr),
                                       static_cast<std::size_t>(cc));
                if (std::isnan(zj)) continue;

                if (zj > zi) is_max = false;
                if (zj < zi) state[j] = NLM;  // CHM ws is uniform → always cascade
                if (zj == zi && lm_grid[j]) return false;
            }
        }
        return is_max;
    };

    for (std::size_t r = 0; r < H; ++r) {
        for (std::size_t c = 0; c < W; ++c) {
            if (state[idx(r, c)] == NLM) continue;
            if (is_local_max(r, c)) {
                lm_grid[idx(r, c)] = 1;
                rows.push_back(static_cast<std::int32_t>(r));
                cols.push_back(static_cast<std::int32_t>(c));
            }
        }
    }
}

}  // namespace pylidar::core::its
