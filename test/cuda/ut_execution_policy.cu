#include <gtest/gtest.h>
#include <mtensor.hpp>

using namespace matazure;
using namespace testing;

__global__ void test_kernel() {}

TEST(CudaExecutionTest, ExecutionPolicy) {
    {
        cuda::execution_policy policy;
        cuda::configure_grid(policy, test_kernel);
        std::cout << "grid dim " << policy.grid_dim() << std::endl;
        std::cout << "block dim " << policy.block_dim() << std::endl;
        std::cout << "shared memory bytes " << policy.shared_mem_bytes() << std::endl;
    }
}

TEST(CudaExecutionTest, DefaultExecutionPolicy) {
    {
        cuda::default_execution_policy policy;
        cuda::configure_grid(policy, test_kernel);
        std::cout << "grid dim " << policy.grid_dim() << std::endl;
        std::cout << "block dim " << policy.block_dim() << std::endl;
        std::cout << "shared memory bytes " << policy.shared_mem_bytes() << std::endl;
    }
}

TEST(CudaExecutionTests, ForIndexExecutionPolicy) {
    {
        cuda::for_index_execution_policy policy;
        policy.total_size(64);
        cuda::configure_grid(policy, test_kernel);
        std::cout << "grid dim " << policy.grid_dim() << std::endl;
        std::cout << "block dim " << policy.block_dim() << std::endl;
        std::cout << "shared memory bytes " << policy.shared_mem_bytes() << std::endl;
    }

    {
        cuda::for_index_execution_policy policy;
        policy.total_size(128);
        cuda::configure_grid(policy, test_kernel);
        std::cout << "grid dim " << policy.grid_dim() << std::endl;
        std::cout << "block dim " << policy.block_dim() << std::endl;
        std::cout << "shared memory bytes " << policy.shared_mem_bytes() << std::endl;
    }

    {
        cuda::for_index_execution_policy policy;
        policy.total_size(0);
        cuda::configure_grid(policy, test_kernel);
        std::cout << "grid dim " << policy.grid_dim() << std::endl;
        std::cout << "block dim " << policy.block_dim() << std::endl;
        std::cout << "shared memory bytes " << policy.shared_mem_bytes() << std::endl;
    }
}
