// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar._core — nanobind extension entry point.
//
// Phase 0 registered set_log_callback only.
// Phase 1 adds smooth_height as the first algorithm binding.
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
#include "common/point_cloud.hpp"
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
    if (shape != 1 && shape != 2) {
        throw std::invalid_argument(
            "smooth_height: shape must be 1 (square) or 2 (circular)");
    }

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
}
