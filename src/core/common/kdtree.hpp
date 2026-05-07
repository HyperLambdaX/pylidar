#pragma once

// Thin header-only wrapper around vendored nanoflann (BSD-2,
// src/third_party/nanoflann/) for the 2D xy queries M2 algorithms need.
// Only what the port actually consumes is exposed:
//   - 2D KD-tree built on an (N, 2) double buffer
//   - radial search (squared L2 distance ≤ r²)
// Square-window queries are realized by callers as "circle of bbox-radius
// then filter to box"; the wrapper deliberately stays radial-only so we
// don't reinvent nanoflann's internal box-search machinery.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nanoflann.h"

namespace pylidar::core {

// Lifetime contract: KdTree2D borrows the (n, 2) xy buffer it is
// constructed with — it does NOT copy the data. The caller is responsible
// for keeping the buffer alive at least as long as the KdTree2D instance.
// Mutating the buffer after construction also invalidates the index
// (nanoflann does no defensive copy). For ITS algorithms in this repo the
// tree's lifetime is bounded by a single bindings entry call, so the
// caller-side lifetime is trivially satisfied by the local std::vector
// the binding builds before constructing the tree.
class KdTree2D {
public:
    using IndexType = std::uint32_t;

    // xy must be a (n, 2) row-major buffer; lifetime must outlive *this.
    KdTree2D(const double* xy, std::size_t n)
        : adaptor_{xy, n},
          index_(2, adaptor_, make_params())
    {
        // nanoflann 1.7+ builds in init() unless SkipInitialBuildIndex set.
    }

    KdTree2D(const KdTree2D&) = delete;
    KdTree2D& operator=(const KdTree2D&) = delete;

    // Returns indices of all points within Euclidean distance r of (cx, cy).
    // out_idx is overwritten. Note: nanoflann radius is squared distance.
    void radius_search(double cx, double cy, double r,
                       std::vector<IndexType>& out_idx) const {
        const double query[2] = {cx, cy};
        std::vector<nanoflann::ResultItem<IndexType, double>> hits;
        index_.radiusSearch(query, r * r, hits);
        out_idx.clear();
        out_idx.reserve(hits.size());
        for (const auto& h : hits) out_idx.push_back(h.first);
    }

private:
    struct Adaptor {
        const double* xy;
        std::size_t n;

        std::size_t kdtree_get_point_count() const { return n; }
        double kdtree_get_pt(std::size_t i, std::size_t d) const {
            return xy[i * 2 + d];
        }
        // We don't precompute a bbox — let nanoflann derive it.
        template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
    };

    using IndexImpl = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, Adaptor>,
        Adaptor, 2, IndexType>;

    static nanoflann::KDTreeSingleIndexAdaptorParams make_params() {
        nanoflann::KDTreeSingleIndexAdaptorParams p;
        // Force single-threaded index build: nanoflann's default reaches for
        // std::thread::hardware_concurrency(), which we don't want pulling in
        // an implicit thread pool for sub-microsecond builds in tests.
        p.n_thread_build = 1;
        return p;
    }

    Adaptor adaptor_;
    IndexImpl index_;
};

}  // namespace pylidar::core
