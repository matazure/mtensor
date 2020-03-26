#include <mtensor.hpp>
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
#define HETE_TENSOR cuda::tensor
#define TAG device_tag
#endif

#ifdef USE_HOST
#define HETE_TENSOR tensor
#define TAG host_tag
#endif

#ifdef USE_OPENCL
#error "not support"
#endif

typedef Types<HETE_TENSOR<int, 1>, HETE_TENSOR<int, 2>, HETE_TENSOR<int, 3>, HETE_TENSOR<int, 4>> ImplementTypes;

TYPED_TEST_CASE(CommonTest, ImplementTypes);

TYPED_TEST(CommonTest, TestStride){

}
