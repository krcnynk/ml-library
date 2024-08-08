#include <gtest/gtest.h>
#include "ml_framework/tensor.h"
#include <Eigen/Dense>

// Test tensor initialization and shape
TEST(TensorTest, Initialization)
{
    Eigen::VectorXi shape(2);
    shape << 2, 3;
    ml_framework::Tensor tensor(shape);

    // Check tensor shape
    const Eigen::VectorXi &tensor_shape = tensor.shape();
    ASSERT_EQ(tensor_shape.size(), shape.size());
    for (int i = 0; i < shape.size(); ++i)
    {
        EXPECT_EQ(tensor_shape[i], shape[i]);
    }

    // Check data size
    EXPECT_EQ(tensor.data().size(), 6);
}

// Test tensor addition
TEST(TensorTest, Addition)
{
    Eigen::VectorXi shape(2);
    shape << 2, 3;
    Eigen::VectorXf data1(6);
    data1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Eigen::VectorXf data2(6);
    data2 << 6.0, 5.0, 4.0, 3.0, 2.0, 1.0;

    ml_framework::Tensor tensor1(shape, data1);
    ml_framework::Tensor tensor2(shape, data2);

    ml_framework::Tensor result = tensor1 + tensor2;

    Eigen::VectorXf expected_data(6);
    expected_data << 7.0, 7.0, 7.0, 7.0, 7.0, 7.0;
    EXPECT_TRUE(result.data().isApprox(expected_data));
}

// Test tensor multiplication
TEST(TensorTest, Multiplication)
{
    Eigen::VectorXi shape(2);
    shape << 2, 3;
    Eigen::VectorXf data1(6);
    data1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Eigen::VectorXf data2(6);
    data2 << 6.0, 5.0, 4.0, 3.0, 2.0, 1.0;

    ml_framework::Tensor tensor1(shape, data1);
    ml_framework::Tensor tensor2(shape, data2);

    ml_framework::Tensor result = tensor1 * tensor2;

    Eigen::VectorXf expected_data(6);
    expected_data << 6.0, 10.0, 12.0, 12.0, 10.0, 6.0;
    EXPECT_TRUE(result.data().isApprox(expected_data));
}

// Test tensor data
TEST(TensorTest, Data)
{
    Eigen::VectorXi shape(2);
    shape << 2, 3;
    Eigen::VectorXf data(6);
    data << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    ml_framework::Tensor tensor(shape, data);

    const Eigen::VectorXf &tensor_data = tensor.data();
    ASSERT_EQ(tensor_data.size(), data.size());
    EXPECT_TRUE(tensor_data.isApprox(data));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
