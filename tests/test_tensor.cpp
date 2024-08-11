#include <gtest/gtest.h>
#include "tensor.h"

namespace ml_framework {

class TensorTest : public ::testing::Test {
protected:
    // Setup function to run before each test
    void SetUp() override {
        // Code to initialize Tensor objects or other setup operations
    }

    // Teardown function to run after each test
    void TearDown() override {
        // Code to clean up after tests
    }
};

// Test for Tensor constructor with shape
TEST_F(TensorTest, ConstructorWithShape) {
    std::vector<size_t> shape = {2, 3};
    Tensor tensor(shape);
    EXPECT_EQ(static_cast<const ml_framework::Tensor*>(&tensor)->shape(), shape);
}

// // Test for Tensor constructor with shape and data
// TEST_F(TensorTest, ConstructorWithShapeAndData) {
//     std::vector<size_t> shape = {2, 3};
//     std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
//     Tensor tensor(shape, data.data());
//     EXPECT_EQ(tensor.shape(), shape);
//     // Add additional checks to verify data
// }

// // Test Tensor addition operator
// TEST_F(TensorTest, AdditionOperator) {
//     std::vector<size_t> shape = {2, 2};
//     std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
//     std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};

//     Tensor tensor1(shape, data1.data());
//     Tensor tensor2(shape, data2.data());
//     Tensor result = tensor1 + tensor2;

//     std::vector<float> expected_data = {6.0f, 8.0f, 10.0f, 12.0f};
//     // Verify result data
// }

// // Test Tensor element-wise multiplication operator
// TEST_F(TensorTest, MultiplicationOperator) {
//     std::vector<size_t> shape = {2, 2};
//     std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
//     std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};

//     Tensor tensor1(shape, data1.data());
//     Tensor tensor2(shape, data2.data());
//     Tensor result = tensor1 * tensor2;

//     std::vector<float> expected_data = {5.0f, 12.0f, 21.0f, 32.0f};
//     // Verify result data
// }

// // Test Tensor matrix multiplication
// TEST_F(TensorTest, MatMul) {
//     std::vector<size_t> shape1 = {2, 3};
//     std::vector<size_t> shape2 = {3, 2};
//     std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
//     std::vector<float> data2 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

//     Tensor tensor1(shape1, data1.data());
//     Tensor tensor2(shape2, data2.data());
//     Tensor result = tensor1.matmul(tensor2);

//     std::vector<float> expected_data = {58.0f, 64.0f, 139.0f, 154.0f};
//     // Verify result data
// }

// Test Tensor destructor
TEST_F(TensorTest, Destructor) {
    {
        Tensor tensor({2, 2});
        // Check that resources are properly cleaned up
    }
    // After this block, the tensor destructor should be called
    // Verify if necessary
}

} // namespace ml_framework

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}