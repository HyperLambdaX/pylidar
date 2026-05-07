#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "dalponte2016.hpp"
#include "matrix2d.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using Float64Mat = nb::ndarray<const double, nb::numpy, nb::ndim<2>, nb::c_contig>;
using Int32Mat   = nb::ndarray<const int32_t, nb::numpy, nb::ndim<2>, nb::c_contig>;
using Int32Out   = nb::ndarray<int32_t, nb::numpy, nb::ndim<2>, nb::c_contig>;

namespace {

Int32Out dalponte2016_binding(Float64Mat chm,
                              Int32Mat seeds,
                              double th_tree,
                              double th_seed,
                              double th_cr,
                              double max_cr) {
    if (chm.shape(0) != seeds.shape(0) || chm.shape(1) != seeds.shape(1)) {
        throw std::invalid_argument(
            "dalponte2016: chm and seeds must share shape");
    }
    if (!(th_seed >= 0.0 && th_seed <= 1.0)) {
        throw std::invalid_argument(
            "dalponte2016: th_seed must be in [0, 1]");
    }
    if (!(th_cr >= 0.0 && th_cr <= 1.0)) {
        throw std::invalid_argument(
            "dalponte2016: th_cr must be in [0, 1]");
    }

    const std::size_t H = chm.shape(0);
    const std::size_t W = chm.shape(1);

    // Allocate result, prefill with seeds, hand ownership to a numpy capsule.
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
    return Int32Out(out, 2, shape, owner);
}

}  // namespace

NB_MODULE(_core, m) {
    m.attr("__version__") = "0.1.0";

    m.def("ping", []() -> const char* { return "pong"; },
          "Health check; returns the literal string \"pong\".");

    m.def("dalponte2016", &dalponte2016_binding,
          nb::kw_only(),
          // .noconvert() refuses implicit float32→float64 / int64→int32 etc.
          // promotions; users must hand us the exact dtypes spec §3 mandates.
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
}
