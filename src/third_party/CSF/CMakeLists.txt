add_library(pylidar_third_party_csf STATIC
    src/c2cdist.cpp
    src/Cloth.cpp
    src/Constraint.cpp
    src/CSF.cpp
    src/Particle.cpp
    src/point_cloud.cpp
    src/Rasterization.cpp
    src/XYZReader.cpp
)

target_include_directories(
    pylidar_third_party_csf
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pylidar_third_party_csf PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(pylidar_third_party_csf PUBLIC CSF_USE_OPENMP=1)
endif()
