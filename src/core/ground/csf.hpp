#pragma once

#include <vector>

#include "point.hpp"

namespace pylidar::core::ground {

struct CsfParams {
    bool slope_smooth = false;
    double class_threshold = 0.5;
    double cloth_resolution = 0.5;
    int rigidness = 1;
    int iterations = 500;
    double time_step = 0.65;
};

std::vector<char> csf_ground(const std::vector<PointXYZ>& pts,
                             const std::vector<char>& candidate,
                             const CsfParams& params);

}  // namespace pylidar::core::ground
