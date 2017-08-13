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
	//if the shape is less than zero, throw invalid_shape exception
	EXPECT_THROW({
		pointi<TypeParam::rank> ext{};
		fill(ext, -10);
		TypeParam ts(ext);
	}, invalid_shape);	

	//the zero shape is valid
	{
		pointi<TypeParam::rank> ext{};
		fill(ext, 0);
		TypeParam ts(ext);
	}

	{
		pointi<TypeParam::rank> ext{};
		fill(ext, 100);
		TypeParam ts(ext);
	}

	{
		pointi<TypeParam::rank> ext{};
		fill(ext, 100);
		auto ts = make_tensor(ext, typename TypeParam::value_type(0.0f),aligned{}, 16);
	}
}

TYPED_TEST(TensorTest, ShapeSize) {
	{
		pointi<TypeParam::rank> ext{};
		fill(ext, 0);
		TypeParam ts(ext);
		
		EXPECT_EQ(0, ts.size());
		for_each(ts.shape(), [](int_t dim) {
			EXPECT_EQ(0, dim);
		});
	}

	{
		pointi<TypeParam::rank> ext{};
		fill(ext, 10);
		TypeParam ts(ext);

		auto expect_size = reduce(ext, 1, [](int_t x1, int_t x2) { return x1*x2; });
		EXPECT_EQ(expect_size, ts.size());
		for_each(ts.shape(), [](int_t dim) {
			EXPECT_EQ(10, dim);
		});
	}

}

TYPED_TEST(TensorTest, Access) {

}
