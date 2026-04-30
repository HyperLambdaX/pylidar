// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar._core — nanobind extension entry point.
//
// Phase 0 only registers `set_log_callback` to validate the binding chain.
// Algorithm functions (smooth_height, lmf_*, segment_*) land in subsequent
// phases by including the relevant `core/its/*.hpp` and adding nb::def calls.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <memory>
#include <string>
#include <string_view>

#include "../core/common/log.hpp"

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

}  // namespace

NB_MODULE(_core, m) {
    m.doc() =
        "pylidar._core — C++ algorithm core (Phase 0: smoke / log only). "
        "Public Python API lives in pylidar.* — do not import _core directly.";

    m.def(
        "set_log_callback",
        &install_python_log_callback,
        nb::arg("callback").none(),
        "Install a Python callable to receive log messages from the C++ core. "
        "Pass None to disable logging (default). The callable is invoked with "
        "a single str argument. Exceptions are routed through "
        "sys.unraisablehook.");
}
