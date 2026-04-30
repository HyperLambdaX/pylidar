// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — nanoflann adaptor for PointCloudXYZ.
//
// Lets nanoflann index a non-owning PointCloudXYZ view directly. dim=2
// (KDTree2D) drops the z coordinate; dim=3 (KDTree3D) uses all three.

#pragma once

#include "../third_party/nanoflann.hpp"
#include "point_cloud.hpp"

#include <cstddef>

namespace pylidar::common {

struct PointCloudXYZ_KDAdaptor {
    const PointCloudXYZ& pc;

    explicit PointCloudXYZ_KDAdaptor(const PointCloudXYZ& p) noexcept : pc(p) {}

    inline std::size_t kdtree_get_point_count() const { return pc.n; }

    inline double kdtree_get_pt(std::size_t idx, std::size_t dim) const {
        if (dim == 0) return pc.x[idx * pc.stride];
        if (dim == 1) return pc.y[idx * pc.stride];
        return pc.z[idx * pc.stride];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bbox*/) const { return false; }
};

using KDTree2D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudXYZ_KDAdaptor>,
    PointCloudXYZ_KDAdaptor,
    2 /* dim — XY only */>;

using KDTree3D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudXYZ_KDAdaptor>,
    PointCloudXYZ_KDAdaptor,
    3>;

}  // namespace pylidar::common
