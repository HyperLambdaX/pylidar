# pylidar._core — nanobind extension. v0.1 binds only set_log_callback to
# validate the build chain end-to-end (CMake → C++ → nanobind → wheel → import).
#
# Algorithm bindings will be added module.cpp in subsequent phases. For v0.2+
# we may split into module_its.cpp / module_dtm.cpp / ...

nanobind_add_module(_core
    NB_STATIC
    module.cpp
)

target_link_libraries(_core PRIVATE pylidar::core pylidar::its)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# The wheel is built with `wheel.packages = ["python/pylidar"]`, so the
# extension lands inside the pylidar package directory.
install(TARGETS _core LIBRARY DESTINATION pylidar)
