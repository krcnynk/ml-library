# Define the CUDA source files
set(CUDA_SOURCES kernel.cu)

# Object library
add_library(cuda_objs OBJECT ${CUDA_SOURCES})

set(CUDA_ARCHITECTURES 41234)
# executable target, library target etc.
set_target_properties(cuda_objs PROPERTIES
    CUDA_STANDARD 20  # Set this according to your requirements
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)

add_library(ml_framework STATIC
    tensor.cpp
    autograd.cpp
    module.cpp
    linear.cpp
    relu.cpp
    mse_loss.cpp
    utils.cpp
    $<TARGET_OBJECTS:cuda_objs>
)
set(MY_DEBUG_OPTIONS "-ggdb;-O0;-pedantic-errors;-Wall;-Weffc++;-Wextra;-Wconversion;-Wsign-conversion;")
target_compile_options(ml_framework PRIVATE ${MY_DEBUG_OPTIONS})

set_target_properties(ml_framework PROPERTIES
    CMAKE_CXX_STANDARD 20  # Set this according to your requirements
    CMAKE_CXX_STANDARD_REQUIRED True
    CMAKE_CXX_EXTENSIONS OFF
    CUDA_ARCHITECTURES 75
)

# target_include_directories(ml_framework PRIVATE
#     ${CMAKE_SOURCE_DIR}/include/ml_framework  # Path to the headers
# )

# Link CUDA libraries
target_link_libraries(ml_framework PRIVATE CUDA::cudart CUDA::cublas)

# Include directories for Eigen and any other headers
# target_include_directories(ml_framework PRIVATE ${EIGEN3_INCLUDE_DIR})
# target_include_directories(ml_framework PRIVATE ${CUDA_INCLUDE_DIRS})

