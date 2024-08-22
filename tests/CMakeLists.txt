# Add test source files
set(TEST_SOURCES
    test_tensor.cpp
    # test_linear.cpp
    # test_module.cpp
    test_autograd.cpp
    main.cpp
    # Add more test files here
)

# Create test executable
add_executable(my_cuda_tests ${TEST_SOURCES})


target_link_libraries(my_cuda_tests PRIVATE ml_framework GTest::gtest_main)

# Link the library
# target_link_libraries(my_cuda_tests PRIVATE ml_framework)