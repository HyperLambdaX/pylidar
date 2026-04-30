# pylidar::its — individual tree segmentation algorithm family.
#
# Phase 1 turns this from an INTERFACE placeholder into a STATIC lib now that
# smooth_height.cpp introduces the first translation unit. Future ITS
# algorithms (lmf, dalponte2016, silva2016, li2012) just append their .cpp
# here.

add_library(pylidar_its STATIC
    smooth_height.cpp
)
add_library(pylidar::its ALIAS pylidar_its)

target_link_libraries(pylidar_its PUBLIC pylidar::core)

# OpenMP is required at the top-level CMakeLists; link it on the algorithm lib
# so #pragma omp inside .cpp files actually parallelises.
if(OpenMP_CXX_FOUND)
    target_link_libraries(pylidar_its PUBLIC OpenMP::OpenMP_CXX)
endif()
