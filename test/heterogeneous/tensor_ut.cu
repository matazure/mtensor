#include <matazure/tensor>
#include <gtest/gtest.h>

using namespace matazure;
using namespace testing;

///don't anythiny in the TensorTest, it's just a helper class
template <typename _Type>
class TensorTest: public testing::Test { };

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

TYPED_TEST_CASE(TensorTest, ImplementTypes);

TYPED_TEST(TensorTest, TestConstruct){
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
	}
}

TYPED_TEST(TensorTest, TestShapeAndSize) {
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

TYPED_TEST(TensorTest, TestAccess) {
	pointi<TypeParam::rank> ext{};
	fill(ext, 10);
	TypeParam ts(ext);
	auto zero = matazure::zero<typename TypeParam::value_type>::value();
	fill(ts, zero);

#ifdef USE_HOST
	auto ts_check = ts;
#else
	auto ts_check = mem_clone(ts, host_tag{});
#endif

	for_each(ts_check, [zero] (const typename TypeParam::value_type &e) {
		EXPECT_TRUE(e == zero);
	});
}
