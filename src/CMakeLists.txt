add_library(ml_framework STATIC
    tensor.cpp
    autograd.cpp
    module.cpp
    linear.cpp
    relu.cpp
    mse_loss.cpp
    utils.cpp
)

target_include_directories(ml_framework PRIVATE
    ${CMAKE_SOURCE_DIR}/include/ml_framework  # Path to the headers
)

# Link CUDA libraries
target_link_libraries(ml_framework PRIVATE CUDA::cudart CUDA::cublas)
# Link Eigen
target_link_libraries(ml_framework PRIVATE Eigen3::Eigen)

# Include directories for Eigen and any other headers
# target_include_directories(ml_framework PRIVATE ${EIGEN3_INCLUDE_DIR})
# target_include_directories(ml_framework PRIVATE ${CUDA_INCLUDE_DIRS})