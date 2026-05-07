cmake_minimum_required(VERSION 3.26)
project(pylidar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# Note: -Werror is intentionally NOT enabled (nanobind template macros trip
# some -Wpedantic warnings). -ffast-math is also NOT enabled (it breaks
# std::isnan semantics, which the ITS algorithms depend on).

find_package(OpenMP COMPONENTS CXX QUIET)   # found-but-not-required; this
                                            # milestone has zero `#pragma omp`.

find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(src/core)
add_subdirectory(src/bindings)
