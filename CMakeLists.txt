cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER "g++")
# Sets the name of the project, and stores it in the variable PROJECT_NAME. 
# When called from the top-level CMakeLists.txt also stores the project name 
# in the variable CMAKE_PROJECT_NAME.
project(MLFramework LANGUAGES C CXX CUDA)

# set variable value
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CUDA_STANDARD 20)  # or another standard if needed
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# add_compile_options(${MY_DEBUG_OPTIONS})

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

message(STATUS "Available CUDA targets:")
message(STATUS "CUDA::cudart: ${CUDA_cudart_LIBRARY}")
message(STATUS "CUDA::cublas: ${CUDA_cublas_LIBRARY}")
message(STATUS "CUDA Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "${CMAKE_SOURCE_DIR}")

# included HEADERS
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(include/ml_framework)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Enable testing
add_subdirectory(src)
add_subdirectory(tests)

# cmake -DCMAKE_BUILD_TYPE=Debug ..