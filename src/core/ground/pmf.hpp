#pragma once

#include <vector>

#include "point.hpp"

namespace pylidar::core::ground {

std::vector<char> pmf_ground(const std::vector<PointXYZ>& pts,
                             const std::vector<double>& ws,
                             const std::vector<double>& th,
                             const std::vector<char>& candidate);

}  // namespace pylidar::core::ground
