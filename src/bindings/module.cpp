#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_core, m) {
    m.attr("__version__") = "0.1.0";
    m.def("ping", []() -> const char* { return "pong"; },
          "Health check; returns the literal string \"pong\".");
}
