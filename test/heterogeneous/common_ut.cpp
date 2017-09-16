#include <matazure/tensor>
#include <gtest/gtest.h>

using namespace matazure;
using namespace testing;

///don't anythiny in the CommonTest, it's just a helper class
template <typename _Type>
class CommonTest: public testing::Test {
public:

	int a;
};

#ifdef USE_CUDA
#define TENSOR cuda::tensor
#define TAG device_tag
#endif

#ifdef USE_HOST
#define TENSOR tensor
#define TAG host_tag
#endif

#ifdef USE_OPENCL
#error "not support"
#endif

typedef Types<TENSOR<int, 1>, TENSOR<int, 2>, TENSOR<int, 3>, TENSOR<int, 4>> ImplementTypes;

TYPED_TEST_CASE(CommonTest, ImplementTypes);

TYPED_TEST(CommonTest, TestStride){

}
