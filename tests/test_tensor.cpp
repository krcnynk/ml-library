#include <gtest/gtest.h>
#include "ml_framework/tensor.h"

// Test tensor initialization and shape
TEST(TensorTest, Initialization) {
    std::vector<int> shape = {2, 3};
    ml_framework::Tensor tensor(shape);

    // Check tensor shape
    const std::vector<int>& tensor_shape = tensor.shape();
    ASSERT_EQ(tensor_shape.size(), shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        EXPECT_EQ(tensor_shape[i], shape[i]);
    }

    // Check data size
    EXPECT_EQ(const_cast<const ml_framework::Tensor&>(tensor).data().size(), 6);
}

// Test tensor addition
TEST(TensorTest, Addition) {
    std::vector<int> shape = {2, 3};
    ml_framework::Tensor tensor1(shape, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    ml_framework::Tensor tensor2(shape, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    
    ml_framework::Tensor result = tensor1 + tensor2;
    
    std::vector<float> expected_data = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
    EXPECT_EQ(const_cast<const ml_framework::Tensor&>(result).data(), expected_data);
}

// Test tensor multiplication
TEST(TensorTest, Multiplication) {
    std::vector<int> shape = {2, 3};
    ml_framework::Tensor tensor1(shape, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    ml_framework::Tensor tensor2(shape, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    
    ml_framework::Tensor result = tensor1 * tensor2;
    
    std::vector<float> expected_data = {6.0, 10.0, 12.0, 12.0, 10.0, 6.0};
    EXPECT_EQ(const_cast<const ml_framework::Tensor&>(result).data(), expected_data);
}

// Test tensor data
TEST(a, Data) {
    std::vector<int> shape = {2, 3};
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    ml_framework::Tensor tensor(shape, data);

    const std::vector<float>& tensor_data = const_cast<const ml_framework::Tensor&>(tensor).data();
    ASSERT_EQ(tensor_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(tensor_data[i], data[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
