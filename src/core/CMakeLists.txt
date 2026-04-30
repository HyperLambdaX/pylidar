# pylidar_core: the algorithm-family core. Header-only in Phase 0; will become
# a STATIC library as soon as the first algorithm family adds a .cpp.
#
# Architecture rule (spec §5, §12): core/common/ depends on nothing in
# core/<algo>/; each algorithm family lives in its own subdirectory with its
# own CMakeLists.txt.

add_library(pylidar_core INTERFACE)
add_library(pylidar::core ALIAS pylidar_core)

target_include_directories(pylidar_core INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_compile_features(pylidar_core INTERFACE cxx_std_17)

# Algorithm families. v0.1 ships only `its`; future phases add dtm/, metrics/, ...
add_subdirectory(its)
