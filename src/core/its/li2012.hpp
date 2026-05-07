#pragma once

#include <cstdint>
#include <vector>

#include "point.hpp"

namespace pylidar::core::its {

// Tree segmentation per Li, Guo, Jakubowski & Kelly (2012).
//
// Inputs (xyz):
//   pts        : (N,) PointXYZ in arbitrary order
//   dt1, dt2   : distance thresholds (positive, in xy units of pts) — dt1 used
//                while u.z <= Zu, dt2 used when u.z > Zu (Li et al. eq. 5)
//   Zu         : height cutoff between dt1 and dt2 (Li et al.)
//   R          : LMF window radius (xy diameter is 2R) for the local-max pre
//                pass; pass R<=0 to skip (treat all points as local maxima)
//   th_tree    : addition from lidR (not in original Li 2012) — drop the
//                global-highest point's tree if its z < th_tree
//   radius     : optimisation cap (xy distance from the current treetop u);
//                points farther than radius are auto-pushed to N (no Li rules
//                applied). lidR uses this to short-circuit huge clouds; pass
//                a generous value (>= scene diameter) for "no cap"
//
// Output:
//   ids        : (N,) int32, 1-based tree id for each point; 0 marks
//                "unassigned" (lidR uses NA_INTEGER; we use 0 to match
//                pylidar's silva2016/dalponte2016 convention)
//
// Behaviour: 1:1 with lidR/src/LAS.cpp:1113 segment_trees; the pre-pass
// `filter_local_maxima(R, 0, circular=true)` is replicated inline so we don't
// import the LMF translation unit. Distance comparisons are 2D (xy), matching
// `lidR/inst/include/lidR/Point.h::sqdistance`. First pass is O(N²) per spec
// §M2; KD-tree acceleration is a follow-up perf PR.
void li2012(const std::vector<PointXYZ>& pts,
            double dt1,
            double dt2,
            double Zu,
            double R,
            double th_tree,
            double radius,
            std::vector<std::int32_t>& ids);

}  // namespace pylidar::core::its
