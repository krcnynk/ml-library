cmake_minimum_required(VERSION 3.10)
project(MLFramework LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CUDA_STANDARD 14)  # or 11, 17, etc.
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80")
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/lib/nvidia-cuda-toolkit")
# -gdwarf
set(MY_DEBUG_OPTIONS "-ggdb-4;-O0;-pedantic-errors;-Wall;-Weffc++;-Wextra;-Wconversion;-Wsign-conversion;")
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

# cmake -DCMAKE_BUILD_TYPE=Debug ..