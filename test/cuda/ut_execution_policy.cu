#include <gtest/gtest.h>
#include <mtensor.hpp>
// #include <nvfunctional>

using namespace matazure;
using namespace testing;

__global__ void test_kernel() {}

TEST(CudaExecutionTest, ParallelExecutionPolicy) {
    {
        cuda::parallel_execution_policy policy;
        policy.total_size(64);
        cuda::configure_grid(policy, test_kernel);
        auto block_dim = policy.block_dim();
        auto grid_dim = policy.grid_dim();
        printf("grid dim %d\n", grid_dim);
        printf("block dim %d\n", block_dim);
    }

    {
        cuda::parallel_execution_policy policy;
        policy.total_size(128);
        cuda::configure_grid(policy, test_kernel);
        auto block_dim = policy.block_dim();
        auto grid_dim = policy.grid_dim();
        printf("grid dim %d\n", grid_dim);
        printf("block dim %d\n", block_dim);
    }

    {
        cuda::parallel_execution_policy policy;
        policy.total_size(0);
        cuda::configure_grid(policy, test_kernel);
        auto block_dim = policy.block_dim();
        auto grid_dim = policy.grid_dim();
        printf("grid dim %d\n", grid_dim);
        printf("block dim %d\n", block_dim);
    }

    {
        cuda::execution_policy policy;
        cuda::configure_grid(policy, test_kernel);
        auto block_dim = policy.block_dim();
        auto grid_dim = policy.grid_dim();
        printf("grid dim %d\n", grid_dim);
        printf("block dim %d\n", block_dim);
    }
}
