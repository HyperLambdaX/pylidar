add_library(pylidar_core_its STATIC
    dalponte2016.cpp
    li2012.cpp
    lmf.cpp
    chm_smooth.cpp
    watershed_ext.cpp
)
target_link_libraries(pylidar_core_its PUBLIC pylidar_core_common)
target_include_directories(pylidar_core_its PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
  target_link_libraries(pylidar_core_its PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(pylidar_core_its PRIVATE PYLIDAR_HAS_OPENMP=1)
endif()
