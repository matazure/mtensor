// #include <gtest/gtest.h>
// #include <matazure/cuda/lambda_tensor.hpp>
// #include <mtensor.hpp>
// // #include <nvfunctional>

// using namespace matazure;
// using namespace testing;

// __device__ void print(int i) { printf("%d,", i); }

// struct print_op {
//     MATAZURE_GENERAL void operator()(int i) { printf("%d, ", i); }
// };

// // #pragma nv_exec_check_disable
// // MATAZURE_GENERAL void printhd(int i) { print(i); }

// TEST(LauchTest, OnlySupportDeviceLambda) {
//     cuda::launch([] __device__() { /*printf("thread x: %d\n", threadIdx.x);*/ });
// }

// TEST(ForIndexTest, Lambda) {
//     cuda::for_index(0, 10, [] MATAZURE_GENERAL(int i) { printf("%d,", i); });
//     printf("\n");

//     cuda::for_index(0, 10, [] __device__(int i) { printf("%d,", i); });
//     printf("\n");

//     // if use device code in host, will do none
//     matazure::for_index(0, 10, [] __device__(int i) { printf("%d,", i); });
//     printf("\n");

//     // cuda::for_index(0, 10, print_op{});
//     // printf("\n");
// }
