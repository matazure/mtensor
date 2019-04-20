#include <matazure/unify/tensor.hpp>
#include <matazure/common.hpp>

using namespace matazure;

int main(int argc, char *argv[]) {

	tensor<float, 2> ts(pointi<2>{10, 10});

	unify::tensor<float, 2> uts(ts);

	auto lts = matazure::make_lambda(pointi<2>{10, 10}, [](int_t i) {
		return i;
	});
	typedef decltype(lts) lts_type;
	unify::lambda_tensor<lts_type::rank, typename lts_type::functor_type, typename lts_type::layout_type> ults(lts);

#ifdef MATAURE_CUDA
	auto glcuts = matazure::make_lambda(pointi<2>{10, 10}, [] __matazure__ (int_t i) {
		return i;
	});
	typedef decltype(glcuts) glcuts_type;
	unify::lambda_tensor<glcuts_type::rank, typename glcuts_type::functor_type, typename glcuts_type::layout_type> ults(glcuts);
#endif

	printf("zzm\n");
	return 0;
}
