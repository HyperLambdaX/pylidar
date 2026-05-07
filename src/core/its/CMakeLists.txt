add_library(pylidar_core_its STATIC
    dalponte2016.cpp
    # li2012.cpp              # M2
    # lmf.cpp                 # M2
    # chm_smooth.cpp          # M3
)
target_link_libraries(pylidar_core_its PUBLIC pylidar_core_common)
target_include_directories(pylidar_core_its PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
  target_link_libraries(pylidar_core_its PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(pylidar_core_its PRIVATE PYLIDAR_HAS_OPENMP=1)
endif()
