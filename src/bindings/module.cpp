// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar._core — nanobind extension entry point.
//
// Phase 0 registered set_log_callback only.
// Phase 1 added smooth_height — first algorithm binding.
// Phase 2 adds lmf_points and lmf_chm tree-top detectors.
// Public Python API names live in pylidar.* — _core is internal.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/log.hpp"
#include "common/matrix2d.hpp"
#include "common/point_cloud.hpp"
#include "its/lmf.hpp"
#include "its/shape.hpp"
#include "its/smooth_height.hpp"

namespace nb = nanobind;

namespace {

// Install a Python callable as the C++ log sink. Accepts any Python callable
// or None; raises TypeError on anything else. The wrapper re-acquires the GIL
// before invoking so it stays correct when algorithms run with the GIL
// released.
void install_python_log_callback(nb::object cb) {
    if (cb.is_none()) {
        pylidar::common::log::set_callback({});
        return;
    }
    if (!PyCallable_Check(cb.ptr())) {
        throw nb::type_error("callback must be a callable or None");
    }

    // shared_ptr keeps the Python reference alive for as long as the
    // std::function copy does, even across nested set_callback swaps.
    auto holder = std::make_shared<nb::object>(std::move(cb));

    pylidar::common::log::set_callback(
        [holder](std::string_view msg) {
            nb::gil_scoped_acquire gil;
            try {
                holder->operator()(std::string(msg));
            } catch (nb::python_error& e) {
                // Bad callback shouldn't kill the algorithm. Route through
                // sys.unraisablehook so the user still sees the failure.
                e.discard_as_unraisable("pylidar.set_log_callback");
            }
        });
}

// Helper: copy a std::vector<double> into a freshly-allocated numpy array.
// The capsule owns the heap buffer and frees it on numpy's gc.
nb::ndarray<nb::numpy, double, nb::ndim<1>> vector_to_numpy_1d(
    std::vector<double>&& src) {
    const std::size_t n = src.size();
    auto* buf = new double[n];
    if (n > 0) {
        std::memcpy(buf, src.data(), n * sizeof(double));
    }
    nb::capsule owner(buf, [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        buf, {n}, std::move(owner));
}

// Helper: pack a list of TreeTops into a fresh (M, 3) row-major float64
// numpy array (columns = x, y, z). The id field is intentionally dropped:
// lmf returns id=0 unconditionally, and dalponte/silva take seeds with
// caller-assigned ids via a separate (M, 4) overload (Phase 3+).
nb::ndarray<nb::numpy, double, nb::ndim<2>> treetops_to_numpy_xyz(
    std::vector<pylidar::common::TreeTop>&& src) {
    const std::size_t m = src.size();
    auto* buf = new double[m * 3];
    for (std::size_t i = 0; i < m; ++i) {
        buf[i * 3 + 0] = src[i].x;
        buf[i * 3 + 1] = src[i].y;
        buf[i * 3 + 2] = src[i].z;
    }
    nb::capsule owner(buf, [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        buf, {m, 3}, std::move(owner));
}

void check_shape_int(int shape) {
    if (shape != 1 && shape != 2) {
        throw std::invalid_argument(
            "shape must be 1 (square) or 2 (circular)");
    }
}

// _core.smooth_height — internal entry point. Public API lives in
// pylidar.segmentation.smooth_height which validates inputs and maps the
// "mean"/"gaussian" + "circular"/"square" strings to these ints.
nb::ndarray<nb::numpy, double, nb::ndim<1>> bind_smooth_height(
    nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>
                xyz,
    double      size,
    int         method,
    int         shape,
    double      sigma) {
    if (method != 1 && method != 2) {
        throw std::invalid_argument(
            "smooth_height: method must be 1 (mean) or 2 (gaussian)");
    }
    check_shape_int(shape);

    const double*     base = xyz.data();
    const std::size_t n    = xyz.shape(0);

    // (N,3) row-major numpy → PointCloudXYZ zero-copy view (stride=3).
    pylidar::common::PointCloudXYZ pc{
        base, base + 1, base + 2, n, /*stride=*/3};

    std::vector<double> result;
    {
        nb::gil_scoped_release release;
        result = pylidar::its::smooth_height(
            pc, size,
            static_cast<pylidar::its::SmoothMethod>(method),
            static_cast<pylidar::its::Shape>(shape),
            sigma);
    }
    return vector_to_numpy_1d(std::move(result));
}

// _core.lmf_points — internal entry point. Public API:
// pylidar.locate_trees_lmf_points (segmentation.py), which validates inputs
// and maps "circular"/"square" → 2/1.
nb::ndarray<nb::numpy, double, nb::ndim<2>> bind_lmf_points(
    nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>
                xyz,
    double      ws,
    double      hmin,
    int         shape) {
    check_shape_int(shape);

    const double*     base = xyz.data();
    const std::size_t n    = xyz.shape(0);

    pylidar::common::PointCloudXYZ pc{
        base, base + 1, base + 2, n, /*stride=*/3};

    std::vector<pylidar::common::TreeTop> tops;
    {
        nb::gil_scoped_release release;
        tops = pylidar::its::lmf_points(
            pc, ws, hmin,
            static_cast<pylidar::its::Shape>(shape));
    }
    return treetops_to_numpy_xyz(std::move(tops));
}

// _core.lmf_chm — internal entry point. Public API:
// pylidar.locate_trees_lmf_chm. Caller passes the CHM as a row-major
// (H, W) float64 array plus an unpacked (origin_x, origin_y, pixel_size)
// triple; we transpose into a column-major Matrix2D<double> here, which
// is the convention shared with future Dalponte / Silva CHM algorithms
// (spec §8.1: one O(H·W) memcpy is cheap relative to algorithm cost,
// and lets the C++ algorithm code stay line-aligned with lidR).
nb::ndarray<nb::numpy, double, nb::ndim<2>> bind_lmf_chm(
    nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>
                chm,
    double      origin_x,
    double      origin_y,
    double      pixel_size,
    double      ws,
    double      hmin,
    int         shape) {
    check_shape_int(shape);

    const std::size_t H = chm.shape(0);
    const std::size_t W = chm.shape(1);
    const double*     src = chm.data();

    pylidar::common::RasterView<double> rv;
    rv.data       = pylidar::common::Matrix2D<double>(H, W);
    rv.origin_x   = origin_x;
    rv.origin_y   = origin_y;
    rv.pixel_size = pixel_size;

    // Row-major numpy → column-major Matrix2D, single O(H·W) pass.
    for (std::size_t r = 0; r < H; ++r) {
        const double* row_ptr = src + r * W;
        for (std::size_t c = 0; c < W; ++c) {
            rv.data.at(r, c) = row_ptr[c];
        }
    }

    std::vector<pylidar::common::TreeTop> tops;
    {
        nb::gil_scoped_release release;
        tops = pylidar::its::lmf_chm(
            rv, ws, hmin,
            static_cast<pylidar::its::Shape>(shape));
    }
    return treetops_to_numpy_xyz(std::move(tops));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() =
        "pylidar._core — C++ algorithm core. Public Python API lives in "
        "pylidar.* — do not import _core directly.";

    m.def(
        "set_log_callback",
        &install_python_log_callback,
        nb::arg("callback").none(),
        "Install a Python callable to receive log messages from the C++ core. "
        "Pass None to disable logging (default). The callable is invoked with "
        "a single str argument. Exceptions are routed through "
        "sys.unraisablehook.");

    m.def(
        "smooth_height",
        &bind_smooth_height,
        nb::arg("xyz"),
        nb::arg("size"),
        nb::arg("method"),
        nb::arg("shape"),
        nb::arg("sigma"),
        "Internal: smooth point-cloud Z values. Use pylidar.smooth_height "
        "for the validating string-based API.");

    m.def(
        "lmf_points",
        &bind_lmf_points,
        nb::arg("xyz"),
        nb::arg("ws"),
        nb::arg("hmin"),
        nb::arg("shape"),
        "Internal: local-maximum-filter tree-top detection on an (N,3) "
        "point cloud. Use pylidar.locate_trees_lmf_points instead.");

    m.def(
        "lmf_chm",
        &bind_lmf_chm,
        nb::arg("chm"),
        nb::arg("origin_x"),
        nb::arg("origin_y"),
        nb::arg("pixel_size"),
        nb::arg("ws"),
        nb::arg("hmin"),
        nb::arg("shape"),
        "Internal: local-maximum-filter tree-top detection on a CHM "
        "raster. Use pylidar.locate_trees_lmf_chm instead.");
}
