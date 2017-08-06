#include <matazure/tensor>
#include <gtest/gtest.h>

using namespace matazure;
using namespace testing;

template <typename _Type>
class TensorTest: public testing::Test{
protected:

public:
	typedef _Type tensor_type;
};

#ifdef USE_CUDA
#define TENSOR cuda::tensor
#endif

#ifdef USE_HOST
#define TENSOR tensor
#endif

#ifdef USE_OPENCL
#error "not support"
#endif

typedef Types<TENSOR<int, 1>, TENSOR<int, 2>, TENSOR<int, 3>, TENSOR<int, 4>> ImplementTypes;

TYPED_TEST_CASE(TensorTest, ImplementTypes);

TYPED_TEST(TensorTest, Construct){
	{
		pointi<TypeParam::rank> ext{};
		fill(ext, -10);
		TypeParam ts(ext);
	}

	{
		pointi<TypeParam::rank> ext{};
		fill(ext, 0);
		TypeParam ts(ext);
	}

	{
		pointi<TypeParam::rank> ext{};
		fill(ext, 32);
		TypeParam ts(ext);
	}
}

TYPED_TEST(TensorTest, ShapeStrideSize) {

}

TYPED_TEST(TensorTest, Access) {

}
