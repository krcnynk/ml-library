add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor ml_framework GTest::GTest GTest::Main)

add_executable(test_autograd test_autograd.cpp)
target_link_libraries(test_autograd ml_framework GTest::GTest GTest::Main)

add_executable(test_module test_module.cpp)
target_link_libraries(test_module ml_framework GTest::GTest GTest::Main)

add_executable(test_linear test_linear.cpp)
target_link_libraries(test_linear ml_framework GTest::GTest GTest::Main)

add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME AutogradTest COMMAND test_autograd)
add_test(NAME ModuleTest COMMAND test_module)
add_test(NAME LinearTest COMMAND test_linear)
