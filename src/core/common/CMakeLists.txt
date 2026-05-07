add_library(pylidar_core_common INTERFACE)
target_include_directories(pylidar_core_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Vendored nanoflann goes on the SYSTEM include path so its BSD-2 header is
# exempt from our project -Wpedantic / -Wall regimen (it trips on a few
# variadic-macro and unused-template warnings we don't own).
target_include_directories(pylidar_core_common SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/src/third_party/nanoflann
)
target_compile_features(pylidar_core_common INTERFACE cxx_std_17)
