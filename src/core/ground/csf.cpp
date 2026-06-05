// Wrapper around vendored jianboqi/CSF (Apache-2.0).
//
// lidR's csf() path delegates to the external RCSF package rather than a
// built-in lidR C++ kernel. pylidar mirrors that by calling the official CSF
// C++ implementation through this small adapter.

#include "csf.hpp"

#include <climits>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "CSF.h"

namespace pylidar::core::ground {

namespace {

// Vendored CSF logs progress to std::cout. We redirect it to a sink for the
// duration of do_filtering so the chatter never reaches the host process.
//
// PORT NOTE / precondition: this swaps the *global* std::cout stream buffer, so
// it is only safe when at most one csf_ground() runs at a time within a
// process. pylidar's parallel catalog path (LAScatalog.map_tiles) uses joblib's
// default process-based backend, where each worker owns its own std::cout, so
// the swap is process-local and safe. It is NOT safe under in-process thread
// parallelism — do not call csf_ground concurrently from multiple threads of
// one process. We deliberately leave std::cerr untouched (the vendored CSF
// never writes to it) so any genuine runtime error still surfaces.
class StreamSilencer {
public:
    StreamSilencer() : cout_buf_(std::cout.rdbuf(sink_.rdbuf())) {}

    ~StreamSilencer() {
        std::cout.rdbuf(cout_buf_);
    }

    StreamSilencer(const StreamSilencer&) = delete;
    StreamSilencer& operator=(const StreamSilencer&) = delete;

private:
    std::ostringstream sink_;
    std::streambuf* cout_buf_;
};

}  // namespace

std::vector<char> csf_ground(const std::vector<PointXYZ>& pts,
                             const std::vector<char>& candidate,
                             const CsfParams& params) {
    const std::size_t n = pts.size();
    if (candidate.size() != n) {
        throw std::invalid_argument("csf_ground: candidate length mismatch");
    }

    std::vector<char> out(n, 0);
    std::vector<std::size_t> original_ids;
    original_ids.reserve(n);
    std::vector<double> xyz;
    xyz.reserve(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        if (!candidate[i]) continue;
        original_ids.push_back(i);
        xyz.push_back(pts[i].x);
        xyz.push_back(pts[i].y);
        xyz.push_back(pts[i].z);
    }
    if (original_ids.empty()) return out;
    if (original_ids.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("csf_ground: too many candidate points");
    }

    CSF csf;
    csf.params.bSloopSmooth = params.slope_smooth;
    csf.params.class_threshold = params.class_threshold;
    csf.params.cloth_resolution = params.cloth_resolution;
    csf.params.rigidness = params.rigidness;
    csf.params.interations = params.iterations;
    csf.params.time_step = params.time_step;

    csf.setPointCloud(
        xyz.data(),
        static_cast<int>(original_ids.size()),
        3
    );

    std::vector<int> ground;
    std::vector<int> non_ground;
    {
        StreamSilencer silencer;
        csf.do_filtering(ground, non_ground, false);
    }

    for (int local_id : ground) {
        if (local_id < 0) continue;
        const auto idx = static_cast<std::size_t>(local_id);
        if (idx >= original_ids.size()) continue;
        out[original_ids[idx]] = 1;
    }
    return out;
}

}  // namespace pylidar::core::ground
