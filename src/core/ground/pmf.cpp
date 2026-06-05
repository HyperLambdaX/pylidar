// PORT NOTE — porting from lidR
// Source: lidR/src/LAS.cpp:181-240 (`z_open`) and :826-846
// (`filter_progressive_morphology`), plus RcppFunction.cpp:99-103 (`C_pmf`).
// Behaviour:
//   - Candidate points are held in `skip`; false candidates are ignored.
//   - `z_open` builds the spatial index over the current candidate subset.
//   - The opening window is a square of side `resolution`.
//   - Pass 1 is erosion (minimum Z), although lidR comments call it dilate.
//   - Pass 2 is dilation (maximum pass-1 Z), although lidR comments call it
//     erode.
//   - Pass 2 initializes with std::numeric_limits<double>::min(), not
//     lowest(); this is intentionally replicated for 1:1 behaviour.

#include "pmf.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pylidar::core::ground {

namespace {

constexpr double EPSILON = 1e-8;

struct CellKey {
    std::int64_t ix;
    std::int64_t iy;

    bool operator==(const CellKey& other) const {
        return ix == other.ix && iy == other.iy;
    }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& key) const {
        const auto x = static_cast<std::uint64_t>(key.ix);
        const auto y = static_cast<std::uint64_t>(key.iy);
        // SplitMix-inspired integer finalizers. This handles UTM-sized and
        // negative cell coordinates without building a dense grid.
        std::uint64_t h = x + 0x9e3779b97f4a7c15ULL;
        h ^= y + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebULL;
        h ^= h >> 31;
        return static_cast<std::size_t>(h);
    }
};

class SquareWindowIndex {
public:
    SquareWindowIndex(const std::vector<PointXYZ>& pts,
                      const std::vector<char>& skip,
                      double resolution)
        : pts_(pts), cell_size_(resolution / 8.0) {
        buckets_.reserve(pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i) {
            if (!skip[i]) continue;
            auto& bucket = buckets_[cell_for(pts[i].x, pts[i].y)];
            bucket.ids.push_back(i);
            if (pts[i].z < bucket.min_z) bucket.min_z = pts[i].z;
        }
    }

    void update_max_values(const std::vector<double>& values) {
        for (auto& kv : buckets_) {
            auto& bucket = kv.second;
            bucket.max_value = std::numeric_limits<double>::min();
            for (std::size_t id : bucket.ids) {
                if (values[id] > bucket.max_value) bucket.max_value = values[id];
            }
        }
    }

    double query_min_z(double x, double y, double half_res) const {
        double out = std::numeric_limits<double>::max();
        query_cells(x, y, half_res, [&](const Bucket& bucket, bool full) {
            if (full) {
                if (bucket.min_z < out) out = bucket.min_z;
                return;
            }
            for (std::size_t j : bucket.ids) {
                if (!contains(pts_[j].x, pts_[j].y, x, y, half_res)) continue;
                if (pts_[j].z < out) out = pts_[j].z;
            }
        });
        return out;
    }

    double query_max_value(double x, double y, double half_res,
                           const std::vector<double>& values) const {
        double out = std::numeric_limits<double>::min();
        query_cells(x, y, half_res, [&](const Bucket& bucket, bool full) {
            if (full) {
                if (bucket.max_value > out) out = bucket.max_value;
                return;
            }
            for (std::size_t j : bucket.ids) {
                if (!contains(pts_[j].x, pts_[j].y, x, y, half_res)) continue;
                if (values[j] > out) out = values[j];
            }
        });
        return out;
    }

private:
    struct Bucket {
        std::vector<std::size_t> ids;
        double min_z = std::numeric_limits<double>::max();
        double max_value = std::numeric_limits<double>::min();
    };

    template <class Fn>
    void query_cells(double x, double y, double half_res, Fn&& fn) const {
        const double xmin = x - half_res - EPSILON;
        const double xmax = x + half_res + EPSILON;
        const double ymin = y - half_res - EPSILON;
        const double ymax = y + half_res + EPSILON;
        const std::int64_t ix0 = coord_to_cell(xmin);
        const std::int64_t ix1 = coord_to_cell(xmax);
        const std::int64_t iy0 = coord_to_cell(ymin);
        const std::int64_t iy1 = coord_to_cell(ymax);

        for (std::int64_t ix = ix0; ix <= ix1; ++ix) {
            for (std::int64_t iy = iy0; iy <= iy1; ++iy) {
                const auto it = buckets_.find(CellKey{ix, iy});
                if (it == buckets_.end()) continue;
                const double cell_xmin = static_cast<double>(ix) * cell_size_;
                const double cell_xmax = cell_xmin + cell_size_;
                const double cell_ymin = static_cast<double>(iy) * cell_size_;
                const double cell_ymax = cell_ymin + cell_size_;
                const bool full =
                    cell_xmin >= xmin && cell_xmax <= xmax &&
                    cell_ymin >= ymin && cell_ymax <= ymax;
                fn(it->second, full);
            }
        }
    }

    bool contains(double px, double py, double x, double y, double half_res) const {
        const double dx = std::fabs(px - x);
        const double dy = std::fabs(py - y);
        return dx <= half_res + EPSILON && dy <= half_res + EPSILON;
    }

    std::int64_t coord_to_cell(double v) const {
        return static_cast<std::int64_t>(std::floor(v / cell_size_));
    }

    CellKey cell_for(double x, double y) const {
        return CellKey{coord_to_cell(x), coord_to_cell(y)};
    }

    const std::vector<PointXYZ>& pts_;
    double cell_size_;
    std::unordered_map<CellKey, Bucket, CellKeyHash> buckets_;
};

std::vector<double> z_open(const std::vector<PointXYZ>& pts,
                           const std::vector<char>& skip,
                           double resolution) {
    const std::size_t n = pts.size();
    std::vector<double> z_out(n, 0.0);
    if (n == 0) return z_out;

    const double half_res = resolution / 2.0;

    SquareWindowIndex index(pts, skip, resolution);

    for (std::size_t i = 0; i < n; ++i) {
        if (!skip[i]) continue;

        z_out[i] = index.query_min_z(pts[i].x, pts[i].y, half_res);
    }

    const std::vector<double> z_temp = z_out;
    index.update_max_values(z_temp);
    for (std::size_t i = 0; i < n; ++i) {
        if (!skip[i]) continue;

        z_out[i] = index.query_max_value(pts[i].x, pts[i].y, half_res, z_temp);
    }

    return z_out;
}

}  // namespace

std::vector<char> pmf_ground(const std::vector<PointXYZ>& pts,
                             const std::vector<double>& ws,
                             const std::vector<double>& th,
                             const std::vector<char>& candidate) {
    const std::size_t n = pts.size();
    if (candidate.size() != n) {
        throw std::invalid_argument("pmf_ground: candidate length mismatch");
    }
    if (ws.size() != th.size()) {
        throw std::invalid_argument("pmf_ground: ws and th length mismatch");
    }

    std::vector<char> skip = candidate;
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; ++i) z[i] = pts[i].z;

    for (std::size_t i = 0; i < ws.size(); ++i) {
        const std::vector<double> old_z = z;
        z = z_open(pts, skip, ws[i]);

        for (std::size_t j = 0; j < n; ++j) {
            if (!skip[j]) continue;
            skip[j] = (old_z[j] - z[j]) < th[i] ? 1 : 0;
        }
    }

    return skip;
}

}  // namespace pylidar::core::ground
