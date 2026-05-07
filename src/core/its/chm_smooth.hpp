#pragma once

#include <cstddef>
#include <vector>

#include "point.hpp"

namespace pylidar::core::its {

enum class SmoothMethod {
    Average,
    Gaussian,
};

enum class SmoothShape {
    Circular,
    Square,
};

// Point-cloud height smoothing — 1:1 port of lidR z_smooth (LAS.cpp:112).
//
// Inputs:
//   pts    : (N,) PointXYZ
//   size   : full window size (full diameter for circular, full side
//            length for square). Half-extent = size / 2.
//   method : Average → w = 1; Gaussian → w = (1/(2σ²π))·exp(-d²/(2σ²))
//   shape  : Circular → disc of radius size/2; Square → box ±size/2 each
//   sigma  : Gaussian sd (used only when method == Gaussian). Caller is
//            responsible for any defaulting (e.g. size/6 to match lidR R).
//
// Output:
//   z_out  : (N,) float64; z_out[i] = weighted mean of z over neighbours
//            of point i (the point itself included, see PORT NOTE).
//
// Behaviour: matches lidR z_smooth (LAS.cpp:112), sequential. NaN is not
// expected — lidR doesn't NaN-guard either; if a neighbour has NaN z the
// arithmetic propagates a NaN out, same as lidR.
void chm_smooth(const std::vector<PointXYZ>& pts,
                double size,
                SmoothMethod method,
                SmoothShape  shape,
                double sigma,
                std::vector<double>& z_out);

}  // namespace pylidar::core::its
