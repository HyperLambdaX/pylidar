add_library(pylidar_core_ground STATIC
    csf.cpp
    pmf.cpp
)
target_link_libraries(
    pylidar_core_ground
    PUBLIC
        pylidar_core_common
        pylidar_third_party_csf
)
target_include_directories(pylidar_core_ground PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
