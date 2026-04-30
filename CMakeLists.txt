cmake_minimum_required(VERSION 3.26)
project(pylidar LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Python + nanobind ------------------------------------------------------
# scikit-build-core injects a usable Python automatically; for raw cmake
# invocations we still need the explicit search.
find_package(Python 3.10
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_RC)
if(NOT NB_RC EQUAL 0)
    message(FATAL_ERROR
        "Failed to locate nanobind via 'python -m nanobind --cmake_dir'. "
        "Install it with `pip install nanobind` (it is also in build-system.requires).")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ---- OpenMP -----------------------------------------------------------------
# Phase 0 has no #pragma omp code yet, but we wire the dependency now so the
# build profile is honest and macOS libomp / Windows /openmp:llvm issues
# surface here rather than in Phase 1.
#
# Apple Clang doesn't ship OpenMP support out of the box. Apply the standard
# Homebrew libomp shim before find_package so it succeeds in the most common
# macOS dev setup (and matches the cibuildwheel macOS recipe in pyproject.toml).
if(APPLE AND NOT DEFINED OpenMP_CXX_FLAGS)
    set(_libomp_prefix "")
    if(DEFINED ENV{HOMEBREW_PREFIX})
        set(_libomp_prefix "$ENV{HOMEBREW_PREFIX}/opt/libomp")
    endif()
    foreach(_candidate
        "${_libomp_prefix}"
        "/opt/homebrew/opt/libomp"
        "/usr/local/opt/libomp")
        if(_candidate AND EXISTS "${_candidate}/include/omp.h")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_candidate}/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${_candidate}/lib/libomp.dylib")
            message(STATUS "Configured Apple Clang to use libomp at ${_candidate}")
            break()
        endif()
    endforeach()
endif()
find_package(OpenMP REQUIRED)

# ---- Subprojects ------------------------------------------------------------
add_subdirectory(src/core)
add_subdirectory(src/bindings)
