#include <gtest/gtest.h>
#include <matazure/cuda/lambda_tensor.hpp>
#include <mtensor.hpp>
// #include <nvfunctional>

using namespace matazure;
using namespace testing;

struct test_op {
    int __device__ __host__ operator()(pointi<2> idx) const { return 3; }
};

__host__ int geth() { return 2; }

__device__ int getd() { return 3; }

__device__ __host__ int gethd() { return 4; }

#pragma nv_exec_check_disable
__host__ void gethtest() {
    geth();
    getd();
}

#pragma nv_exe_check_disable
__device__ void getdtest() {
    gethd();
    getd();
}

#pragma nv_exec_check_disable
__host__ __device__ void gethdtest() {
    geth();
    getd();
}

TEST(MakeCudaLambdaTensor, TestMakeByLambda) {
    auto clt_test1 = cuda::make_lambda(pointi<2>{1000, 1000},
                                       [] __matazure__(pointi<2> idx) -> int { return idx[0]; })
                         .persist();
    auto clt_test2 = cuda::make_lambda(pointi<2>{1000, 1000}, test_op{}).persist();

    cuda::device_synchronize();

    // compile error!
    // type traits not work
    // auto clt_test3 = cuda::make_lambda(pointi<2>{1000, 1000},
    //                                    [] __device__(pointi<2> idx) -> int { return idx[0]; })
    //                      .persist();

    // nvstd::function<int(void)> t = test_op{};
}
