cmake_minimum_required(VERSION 3.10)
project(MLFramework LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(MY_DEBUG_OPTIONS "-gdwarf-4;-O0;-pedantic-errors;-Wall;-Weffc++;-Wextra;-Wconversion;-Wsign-conversion;")
add_compile_options(${MY_DEBUG_OPTIONS})

# Find Eigen
find_package(Eigen3 3.3 REQUIRED)  # You can specify the minimum required version of Eigen
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

message(STATUS "Available CUDA targets:")
message(STATUS "CUDA::cudart: ${CUDA_cudart_LIBRARY}")
message(STATUS "${CMAKE_SOURCE_DIR}")

# Add include and source directories
#include_directories(${CUDA_cublas_LIBRARY})
#include_directories(${CUDA_LIBRARIES})

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(include/ml_framework)


# Enable testing
add_subdirectory(src)
add_subdirectory(tests)
