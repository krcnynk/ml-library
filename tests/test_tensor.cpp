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
        const std::vector<int> shape{1, 5};
        float *data = new float[shape[1]]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        Tensor tensor(shape, data);
        EXPECT_EQ(static_cast<const ml_framework::Tensor *>(&tensor)->shape(), shape);
        for (int i = 0; i < 5; ++i)
        {
            EXPECT_FLOAT_EQ(tensor.host_data()[i], data[i]);
            EXPECT_NE(tensor.host_data(), data);
        }
        delete[] data;
    }

    TEST_F(TensorTest, ConstructorWithTensor)
    {
        const std::vector<int> shape{1, 5};
        float *data = new float[shape[1]]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        Tensor tensor1(shape, data);
        Tensor tensor2(shape, data);
        for (int i = 0; i < 5; ++i)
        {
            EXPECT_FLOAT_EQ(tensor2.host_data()[i], tensor1.host_data()[i]);
            EXPECT_NE(tensor2.host_data(), tensor1.host_data());
        }

        delete[] data;
    }

    TEST_F(TensorTest, AdditionOperator)
    {
        // Define shapes and data for the tensors
        const std::vector<int> shape{2, 3};
        float *data1 = new float[shape[0] * shape[1]]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float *data2 = new float[shape[0] * shape[1]]{6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

        // Create tensors
        Tensor tensor1(shape, &data1[0]);
        Tensor tensor2(shape, &data2[0]);

        // Perform addition
        auto result_tensor = tensor1 + tensor2;
        // result_tensor = result_tensor_;
        // Tensor result_tensor = result_tensor_;

        // Define the expected result data
        float *expected_data = new float[shape[0] * shape[1]]{7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};

        // Verify the shape
        EXPECT_EQ(result_tensor->shape(), shape);

        // Verify the data
        const float *result_data = result_tensor->host_data();
        for (int i = 0; i < 6; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], expected_data[i]);
        }

        result_tensor->grad = std::make_unique<Tensor>(shape, 1.0f); // Gradient initialized with ones
        result_tensor->backward();
        std::cout << *tensor1.grad << std::endl;

        delete[] data1;
        delete[] data2;
        delete[] expected_data;
    }

    TEST_F(TensorTest, AdditionOperator1Billion)
    {
        // Define shapes and data for the tensors

        const int size = 1'000'000'0;
        const std::vector<int> shape{1, size};
        float *data1 = new float[size]; //{127.15f};
        float *data2 = new float[size]; //{142.38f};

        std::fill(data1, data1 + size, 127.15f);
        std::fill(data2, data2 + size, 142.38f);
        // Create tensors
        Tensor tensor1(shape, data1);
        delete[] data1;
        Tensor tensor2(shape, data2);
        delete[] data2;

        // Perform addition
        auto result_tensor = tensor1 + tensor2;

        // Define the expected result data
        float *expected_data = new float[size]; //{269.53f};
        std::fill(expected_data, expected_data + size, 269.53f);

        // Verify the shape
        EXPECT_EQ(result_tensor->shape(), shape);

        // Verify the data
        const float *result_data = result_tensor->host_data(); // Adjust as needed for your Tensor class
        for (int i = 0; i < size; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], expected_data[i]);
        }

        delete[] expected_data;
    }

    TEST_F(TensorTest, MultiplicationElementWise)
    {
        // Define shapes and data for the tensors
        const std::vector<int> shape{2, 3};
        float *data1 = new float[shape[0] * shape[1]]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float *data2 = new float[shape[0] * shape[1]]{6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

        // Create tensors
        Tensor tensor1(shape, &data1[0]);
        Tensor tensor2(shape, &data2[0]);

        // Perform addition
        auto result_tensor = tensor1 * tensor2;

        // Define the expected result data
        float *expected_data = new float[shape[0] * shape[1]]{6.0f, 10.0f, 12.0f, 12.0f, 10.0f, 6.0f};

        // Verify the shape
        EXPECT_EQ(result_tensor->shape(), shape);

        // Verify the data
        const float *result_data = result_tensor->host_data();
        for (int i = 0; i < 6; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], expected_data[i]);
        }

        result_tensor->grad = std::make_unique<Tensor>(shape, 2.0f); // Gradient initialized with ones
        result_tensor->backward();
        std::cout << *tensor1.grad << std::endl;
        std::cout << *tensor2.grad << std::endl;

        delete[] data1;
        delete[] data2;
        delete[] expected_data;
    }

    // Test for Tensor constructor with shape
    TEST_F(TensorTest, Printing)
    {
        const std::vector<int> shape{5, 10, 2};
        // 5*20
        float *data = new float[shape[0] * shape[1] * shape[2]]{
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        Tensor tensor(shape, data);
        std::cout << tensor << std::endl;
        delete[] data;
    }

    // TEST_F(TensorTest, Matmul)
    // {
    //     // Define shapes and data for the tensors
    //     int rowsA = 2, colsA = 3;
    //     int rowsB = 3, colsB = 2;
    //     const std::vector<int> shape1{rowsA, colsA};
    //     const std::vector<int> shape2{rowsB, colsB};
    //     // Matrix A: 2x3
    //     float *data1 = new float[rowsA * colsA]{
    //         1.0f, 2.0f, 3.0f, // Row 1
    //         4.0f, 5.0f, 6.0f  // Row 2
    //     };

    //     // Matrix B: 3x2
    //     float *data2 = new float[rowsB * colsB]{
    //         7.0f, 8.0f,  // Row 1
    //         9.0f, 10.0f, // Row 2
    //         11.0f, 12.0f // Row 3
    //     };

    //     // Create tensors
    //     Tensor tensor1(shape1, &data1[0]);
    //     Tensor tensor2(shape2, &data2[0]);

    //     std::cout << tensor1 << tensor2 << std::endl;

    //     // Perform addition
    //     Tensor result_tensor = tensor1.matmul(tensor2);

    //     // Define the expected result data
    //     float *expected_data = new float[rowsA * colsB]{
    //         58.0f, 64.0f,  // Row 1
    //         139.0f, 154.0f // Row 2
    //     };
    //     const std::vector<int> shape3{rowsA, colsB};

    //     Tensor expected_tensor(shape3, expected_data);
    //     // Verify the shape
    //     EXPECT_EQ(result_tensor.shape(), shape3);

    //     // Verify the data
    //     EXPECT_TRUE(result_tensor == expected_tensor);
    //     std::cout << result_tensor << std::endl;

    //     delete[] data1;
    //     delete[] data2;
    //     delete[] expected_data;
    // }

    // Test Tensor destructor
    TEST_F(TensorTest, Destructor)
    {
        {
            const std::vector<int> shape{1, 5};
            const float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            Tensor tensor(shape, &data[0]);
            // Check that resources are properly cleaned up
        }
    }
}
