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

typedef Types<tensor<int, 1>, tensor<int, 2>, tensor<int, 3>, cuda::tensor<int, 1>> ImplementTypes;

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



