#include <matazure/tensor>
#include <gtest/gtest.h>

using namespace matazure;
using namespace testing;

#ifndef GTEST_HAS_TYPED_TEST_P
#error "not support"
#endif

template <typename _Type>
class TensorTest: public testing::Test{
protected:

public:
	typedef _Type tensor_type;
};

#ifdef USE_CUDA
#define TENSOR cuda::tensor
#else
#define TENSOR tensor
#endif

typedef Types<TENSOR<int, 1>, TENSOR<int, 2>, TENSOR<int, 3>, TENSOR<int, 4>> ImplementTypes;

TYPED_TEST_CASE(TensorTest, ImplementTypes);

TYPED_TEST(TensorTest, Construct){
	{
		pointi<tensor_type::rank> ext{};
		fill(ext, -10);
		tensor_type ts(ext);
	}

	{
		pointi<tensor_type::rank> ext{};
		fill(ext, 0);
		tensor_type ts(ext);
	}

	{
		pointi<tensor_type::rank> ext{};
		fill(ext, 32);
		tensor_type ts(ext);
	}
}

TYPED_TEST(TensorTest, ShapeStrideSize) {

}

TYPED_TEST(TensorTest, Access) {

}
