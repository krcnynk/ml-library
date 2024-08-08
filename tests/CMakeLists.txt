# Add test source files
set(TEST_SOURCES
    test_tensor.cpp
    test_linear.cpp
    test_module.cpp
    test_autograd.cpp
    # Add more test files here
)

# Create test executable
add_executable(my_cuda_tests ${TEST_SOURCES})

# Link Google Test
target_link_libraries(my_cuda_tests PRIVATE GTest::GTest GTest::Main)

# Link the library
target_link_libraries(my_cuda_tests PRIVATE ml_framework)