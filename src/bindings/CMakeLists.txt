nanobind_add_module(_core
    NB_STATIC
    STABLE_ABI
    module.cpp
)
target_link_libraries(_core PRIVATE pylidar_core_its)
install(TARGETS _core LIBRARY DESTINATION pylidar)
