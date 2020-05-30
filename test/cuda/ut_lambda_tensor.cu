#include <gtest/gtest.h>
#include <matazure/cuda/lambda_tensor.hpp>
#include <mtensor.hpp>
// #include <nvfunctional>

using namespace matazure;
using namespace testing;

struct device_functor {
    __device__ int operator()(pointi<2> idx) const { return 3; }
};

struct general_functor {
    __host__ __device__ int operator()(pointi<2> idx) const { return 3; }
};

struct host_functor {
    __host__ int operator()(pointi<2> idx) const { return 3; }
};

TEST(CudaLambdaTensorTests, MakeByLambdaFunctor) {
    pointi<2> shape{10, 10};
    auto ts_test1 =
        cuda::make_lambda(shape, [] __host__ __device__(pointi<2> idx) -> int { return idx[0]; })
            .persist();

    // Compiler Error, need __host__ __device__
    // auto ts_test2 = cuda::make_lambda(shape,
    //                                    [] __device__(pointi<2> idx) -> int { return idx[0];
    //                                    })
    //                      .persist();

    auto ts_test3 = cuda::make_lambda(shape, device_functor{}).persist();
}

TEST(CudaLambdaTensorTests, MakeByStructFunctor) {
    pointi<2> shape{10, 10};
    auto ts_test1 = cuda::make_lambda(shape, device_functor{}).persist();
    auto ts_test2 = cuda::make_lambda(shape, general_functor{}).persist();

    // auto ts_test3 =
    // cuda::make_lambda(shape, host_functor{}).persist();  // Run Error, unspecial launch failure
}

TEST(CudaLambdaTensorTests, MakeByArrayAndLinearIndex) {
    pointi<2> shape{10, 10};
    auto ts_test1 =
        cuda::make_lambda(shape, [] __host__ __device__(pointi<2> idx) -> int { return idx[0]; })
            .persist();

    auto ts_test2 = cuda::make_lambda(shape, MLAMBDA(int_t i)->int { return i; }).persist();
}