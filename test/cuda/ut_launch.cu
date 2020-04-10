#include <gtest/gtest.h>
#include <matazure/cuda/lambda_tensor.hpp>
#include <mtensor.hpp>
// #include <nvfunctional>

using namespace matazure;
using namespace testing;

__device__ void print(int i) { printf("%d,", i); }

struct host_functor {
    __host__ void operator()() {}
};

struct general_functor {
    __host__ __device__ void operator()() {
        // printf("thread is %d, %d\n", threadIdx.x, threadIdx.y);
    }
};

struct device_functor {
    __device__ void operator()() {}
};

__device__ void device_function_functor() {}
__host__ void host_function_functor() {
    // printf("thread is %d, %d\n", threadIdx.x, threadIdx.y);
}
__device__ __host__ void general_function_functor() {}

TEST(CudaLaunchTests, LambdaFunctor) {
    cuda::launch([] __device__() {});
    cuda::launch([] __host__ __device__() {});  // Compile Warning
    // cuda::launch([]() {}); //Compile Error
}

TEST(CudaLaunchTests, StructFunctor) {
    cuda::launch(device_functor{});
    cuda::launch(general_functor{});
    // cuda::launch(host_functor{}); //Compiler Error
}

TEST(CudaLaunchTests, FunctionFunctor) {
    //  cuda::launch(device_function_functor); //Compile OK, RUN with an illegal memory access
    // cuda::launch(&device_function_functor); //Compile OK, RUN with an illegal memory access
    // cuda::launch(host_function_functor)); //Compile OK, RUN with an illegal memory access
    // cuda::launch(general_function_functor); //Compile OK, RUN with an illegal memory access
}
