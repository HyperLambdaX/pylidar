#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "dalponte2016.hpp"
#include "li2012.hpp"
#include "lmf.hpp"
#include "matrix2d.hpp"
#include "point.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using Float64Mat   = nb::ndarray<const double,  nb::numpy, nb::ndim<2>, nb::c_contig>;
using Float64Vec   = nb::ndarray<const double,  nb::numpy, nb::ndim<1>, nb::c_contig>;
using Int32Mat     = nb::ndarray<const int32_t, nb::numpy, nb::ndim<2>, nb::c_contig>;
using Int32MatOut  = nb::ndarray<int32_t,       nb::numpy, nb::ndim<2>, nb::c_contig>;
using Int32VecOut  = nb::ndarray<int32_t,       nb::numpy, nb::ndim<1>, nb::c_contig>;
using BoolVecOut   = nb::ndarray<bool,          nb::numpy, nb::ndim<1>, nb::c_contig>;

namespace {

// ─────────────────── dalponte2016 ───────────────────
Int32MatOut dalponte2016_binding(Float64Mat chm,
                                 Int32Mat   seeds,
                                 double th_tree,
                                 double th_seed,
                                 double th_cr,
                                 double max_cr) {
    if (chm.shape(0) != seeds.shape(0) || chm.shape(1) != seeds.shape(1)) {
        throw std::invalid_argument(
            "dalponte2016: chm and seeds must share shape");
    }
    if (!(th_seed >= 0.0 && th_seed <= 1.0)) {
        throw std::invalid_argument("dalponte2016: th_seed must be in [0, 1]");
    }
    if (!(th_cr >= 0.0 && th_cr <= 1.0)) {
        throw std::invalid_argument("dalponte2016: th_cr must be in [0, 1]");
    }

    const std::size_t H = chm.shape(0);
    const std::size_t W = chm.shape(1);

    int32_t* out = new int32_t[H * W];
    std::copy(seeds.data(), seeds.data() + H * W, out);

    pylidar::core::Matrix2D<double> chm_view(
        const_cast<double*>(chm.data()), H, W);
    pylidar::core::Matrix2D<int32_t> seeds_view(
        const_cast<int32_t*>(seeds.data()), H, W);
    pylidar::core::Matrix2D<int32_t> regions_view(out, H, W);

    pylidar::core::its::dalponte2016(
        chm_view, seeds_view, regions_view,
        th_tree, th_seed, th_cr, max_cr);

    nb::capsule owner(out, [](void* p) noexcept {
        delete[] static_cast<int32_t*>(p);
    });
    std::size_t shape[2] = {H, W};
    return Int32MatOut(out, 2, shape, owner);
}

// ─────────────────── li2012 ───────────────────
Int32VecOut li2012_binding(Float64Mat xyz,
                           double dt1,
                           double dt2,
                           double Zu,
                           double R,
                           double hmin,
                           double speed_up) {
    if (xyz.shape(1) != 3) {
        throw std::invalid_argument("li2012: xyz must have shape (N, 3)");
    }
    if (!(dt1 > 0.0 && dt2 > 0.0)) {
        throw std::invalid_argument("li2012: dt1 and dt2 must be > 0");
    }
    if (!(R >= 0.0)) {
        throw std::invalid_argument("li2012: R must be >= 0");
    }
    if (!(Zu > 0.0)) {
        throw std::invalid_argument("li2012: Zu must be > 0");
    }
    if (!(hmin > 0.0)) {
        throw std::invalid_argument("li2012: hmin must be > 0");
    }
    if (!(speed_up > 0.0)) {
        throw std::invalid_argument("li2012: speed_up must be > 0");
    }

    const std::size_t N = xyz.shape(0);
    std::vector<pylidar::core::PointXYZ> pts(N);
    for (std::size_t i = 0; i < N; ++i) {
        pts[i] = {xyz.data()[i * 3 + 0],
                  xyz.data()[i * 3 + 1],
                  xyz.data()[i * 3 + 2]};
    }

    std::vector<int32_t> ids;
    pylidar::core::its::li2012(pts, dt1, dt2, Zu, R, hmin, speed_up, ids);

    int32_t* out = new int32_t[N];
    std::copy(ids.begin(), ids.end(), out);

    nb::capsule owner(out, [](void* p) noexcept {
        delete[] static_cast<int32_t*>(p);
    });
    std::size_t shape[1] = {N};
    return Int32VecOut(out, 1, shape, owner);
}

// ─────────────────── lmf_points ───────────────────
pylidar::core::its::LmfShape parse_shape(const std::string& s, const char* fn) {
    if (s == "circular") return pylidar::core::its::LmfShape::Circular;
    if (s == "square")   return pylidar::core::its::LmfShape::Square;
    throw std::invalid_argument(
        std::string(fn) + ": shape must be 'circular' or 'square'");
}

BoolVecOut lmf_points_binding(Float64Mat xyz,
                              Float64Vec ws,
                              double hmin,
                              const std::string& shape_str,
                              bool is_uniform) {
    if (xyz.shape(1) != 3) {
        throw std::invalid_argument("lmf_points: xyz must have shape (N, 3)");
    }
    if (xyz.shape(0) != ws.shape(0)) {
        throw std::invalid_argument(
            "lmf_points: ws length must equal xyz row count");
    }
    const auto shape = parse_shape(shape_str, "lmf_points");

    const std::size_t N = xyz.shape(0);
    std::vector<pylidar::core::PointXYZ> pts(N);
    for (std::size_t i = 0; i < N; ++i) {
        pts[i] = {xyz.data()[i * 3 + 0],
                  xyz.data()[i * 3 + 1],
                  xyz.data()[i * 3 + 2]};
    }
    std::vector<double> ws_vec(ws.data(), ws.data() + N);
    for (double w : ws_vec) {
        if (!(w > 0.0)) {
            throw std::invalid_argument(
                "lmf_points: ws must be strictly positive");
        }
    }

    std::vector<char> lm_char;
    pylidar::core::its::lmf_points(
        pts, ws_vec, hmin, shape, is_uniform, lm_char);

    bool* out = new bool[N];
    for (std::size_t i = 0; i < N; ++i) out[i] = static_cast<bool>(lm_char[i]);

    nb::capsule owner(out, [](void* p) noexcept {
        delete[] static_cast<bool*>(p);
    });
    std::size_t shape_arr[1] = {N};
    return BoolVecOut(out, 1, shape_arr, owner);
}

// ─────────────────── lmf_chm ───────────────────
Int32MatOut lmf_chm_binding(Float64Mat chm,
                            double ws,
                            double hmin,
                            const std::string& shape_str) {
    if (!(ws > 0.0)) {
        throw std::invalid_argument("lmf_chm: ws must be > 0");
    }
    const auto shape = parse_shape(shape_str, "lmf_chm");

    const std::size_t H = chm.shape(0);
    const std::size_t W = chm.shape(1);
    pylidar::core::Matrix2D<double> chm_view(
        const_cast<double*>(chm.data()), H, W);

    std::vector<int32_t> rows, cols;
    pylidar::core::its::lmf_chm(chm_view, ws, hmin, shape, rows, cols);

    const std::size_t K = rows.size();
    int32_t* out = new int32_t[K * 2];
    for (std::size_t i = 0; i < K; ++i) {
        out[i * 2 + 0] = rows[i];
        out[i * 2 + 1] = cols[i];
    }

    nb::capsule owner(out, [](void* p) noexcept {
        delete[] static_cast<int32_t*>(p);
    });
    std::size_t shape_arr[2] = {K, 2};
    return Int32MatOut(out, 2, shape_arr, owner);
}

}  // namespace

NB_MODULE(_core, m) {
    m.attr("__version__") = "0.1.0";

    m.def("ping", []() -> const char* { return "pong"; },
          "Health check; returns the literal string \"pong\".");

    m.def("dalponte2016", &dalponte2016_binding,
          nb::kw_only(),
          "chm"_a.noconvert(),
          "seeds"_a.noconvert(),
          "th_tree"_a = 2.0,
          "th_seed"_a = 0.45,
          "th_cr"_a   = 0.55,
          "max_cr"_a  = 10.0,
          "Region-growing tree-crown segmentation (Dalponte & Coomes 2016).\n"
          "chm:    (H, W) float64, C-contiguous canopy height raster.\n"
          "seeds:  (H, W) int32,   C-contiguous; non-zero entries are tree IDs.\n"
          "Returns: (H, W) int32 region map seeded by `seeds` and grown.");

    m.def("li2012", &li2012_binding,
          nb::kw_only(),
          "xyz"_a.noconvert(),
          "dt1"_a      = 1.5,
          "dt2"_a      = 2.0,
          "Zu"_a       = 15.0,
          "R"_a        = 2.0,
          "hmin"_a     = 2.0,
          "speed_up"_a = 10.0,
          "Tree segmentation (Li, Guo, Jakubowski & Kelly 2012).\n"
          "xyz:    (N, 3) float64, C-contiguous point cloud.\n"
          "Returns: (N,) int32 — 1-based tree id, 0 = unassigned.");

    m.def("lmf_points", &lmf_points_binding,
          nb::kw_only(),
          "xyz"_a.noconvert(),
          "ws"_a.noconvert(),
          "hmin"_a       = 2.0,
          "shape"_a      = "circular",
          "is_uniform"_a = false,
          "Local-maximum filter on a point cloud (lidR `lmf` for LAS).\n"
          "xyz:    (N, 3) float64, C-contiguous.\n"
          "ws:     (N,)   float64 — full window size per point. Bindings\n"
          "        (Python wrapper) expand a scalar or callable into this\n"
          "        length-N array; this entrypoint always takes the array.\n"
          "Returns: (N,) bool.");

    m.def("lmf_chm", &lmf_chm_binding,
          nb::kw_only(),
          "chm"_a.noconvert(),
          "ws"_a,
          "hmin"_a  = 2.0,
          "shape"_a = "circular",
          "Local-maximum filter on a CHM raster.\n"
          "chm:    (H, W) float64, C-contiguous.\n"
          "ws:     window size in **pixel** units (not world).\n"
          "Returns: (K, 2) int32 — (row, col) indices of detected maxima.");
}
