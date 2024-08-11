#include <gtest/gtest.h>
#include "tensor.h"

namespace ml_framework
{

    class TensorTest : public ::testing::Test
    {
    protected:
        // Setup function to run before each test
        void SetUp() override
        {
            // Code to initialize Tensor objects or other setup operations
        }

        // Teardown function to run after each test
        void TearDown() override
        {
            // Code to clean up after tests
        }
    };

    // Test for Tensor constructor with shape
    TEST_F(TensorTest, ConstructorWithShape)
    {
        const std::vector<size_t> shape{1, 5};
        const float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Tensor tensor(shape, &data[0]);
        EXPECT_EQ(static_cast<const ml_framework::Tensor *>(&tensor)->shape(), shape);
        for (size_t i = 0; i < 5; ++i)
        {
            EXPECT_FLOAT_EQ(tensor.host_data()[i], data[i]);
            EXPECT_NE(tensor.host_data(), data);
        }
    }

    TEST_F(TensorTest, ConstructorWithTensor)
    {
        const std::vector<size_t> shape{1, 5};
        const float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Tensor tensor1(shape, &data[0]);
        Tensor tensor2(tensor1);
        for (size_t i = 0; i < 5; ++i)
        {
            EXPECT_FLOAT_EQ(tensor2.host_data()[i], tensor1.host_data()[i]);
            EXPECT_NE(tensor2.host_data(), tensor1.host_data());
        }
    }

    TEST_F(TensorTest, AdditionOperator)
    {
        // Define shapes and data for the tensors
        const std::vector<size_t> shape{2, 3};
        const float data1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const float data2[] = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

        // Create tensors
        Tensor tensor1(shape, &data1[0]);
        Tensor tensor2(shape, &data2[0]);

        // Perform addition
        Tensor result_tensor = tensor1 + tensor2;

        // Define the expected result data
        const float expected_data[] = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};

        // Verify the shape
        EXPECT_EQ(result_tensor.shape(), shape);

        // Verify the data
        const float *result_data = result_tensor.host_data();
        for (size_t i = 0; i < 6; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], expected_data[i]);
        }
    }

    // TEST_F(TensorTest, AdditionOperator1Million)
    // {
    //     // Define shapes and data for the tensors

    //     const size_t size = 1000000;
    //     const std::vector<size_t> shape{1, size};
    //     const float data1[size]{100.15};
    //     const float data1[size]{142.35};

    //     // Create tensors
    //     Tensor tensor1(shape, &data1[0]);
    //     Tensor tensor2(shape, &data2[0]);

    //     // Perform addition
    //     Tensor result_tensor = tensor1 + tensor2;

    //     // Define the expected result data
    //     const float expected_data[] = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};

    //     // Verify the shape
    //     EXPECT_EQ(result_tensor.shape(), shape);

    //     // Verify the data
    //     const float *result_data = result_tensor.host_data(); // Adjust as needed for your Tensor class
    //     for (size_t i = 0; i < 6; ++i)
    //     {
    //         EXPECT_FLOAT_EQ(result_data[i], expected_data[i]);
    //     }
    // }

    // Test Tensor destructor
    TEST_F(TensorTest, Destructor)
    {
        {
            const std::vector<size_t> shape{1,5};
            const float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            Tensor tensor(shape, &data[0]);
            // Check that resources are properly cleaned up
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto retVal = RUN_ALL_TESTS();
    return retVal;
}