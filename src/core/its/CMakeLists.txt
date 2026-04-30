# pylidar::its — individual tree segmentation algorithm family.
#
# Phase 0 has no algorithm sources yet (Phase 1 introduces smooth_height as the
# first .cpp). To avoid an empty library we declare an INTERFACE target that
# Phase 1 will convert to STATIC by appending sources.

add_library(pylidar_its INTERFACE)
add_library(pylidar::its ALIAS pylidar_its)
target_link_libraries(pylidar_its INTERFACE pylidar::core)
