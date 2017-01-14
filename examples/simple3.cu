#include <matazure/tensor>

using namespace matazure;

//nvcc bug
//tensor要在使用make_lambda之前用一次，否则会编译错误，内部已调用常用的tensor
//using tensor2i = tensor<int, 2>;
//using tensor2f = tensor<float, 2>;

int main() {
	//构造一个linear access的二维int型lambda_tensor
	auto lts_linear = make_lambda(pointi<2>{ 4, 4 }, [] (int_t i)->int{
		return i;
	});
	printf("linear access lts_linear.\n");
	for (int_t i = 0; i < lts_linear.size(); ++i) {
		printf("%d: %d\n", i, lts_linear[i]);
	}

	//构造一个array access的二维float型lambda_tensor
	auto  lts_array = make_lambda(pointi<2>{ 4, 4 }, [] (pointi<2> idx)->float{
		return idx[0] + idx[1] * 0.1f;
	});
	printf("array access lts_array.\n");
	//按数组访问的方式打印值
	for (int_t j = 0; j < lts_array.extent()[1]; ++j) {
		for (int_t i = 0; i < lts_array.extent()[0]; ++i) {
			//通过pointi来访问tensor
			printf("(%d, %d): %f\n", i, j, lts_array(pointi<2>{i, j}));
		}
	}

	//固化lambda_tensor
	tensor<float, 2> ts_array = lts_array.persist();
	static_assert(is_same<decltype(ts_array), tensor<float, 2>>::value, "");

	printf("array access ts_array.\n");
	//按数组访问的方式打印值
	for (int_t j = 0; j < ts_array.extent()[1]; ++j) {
		for (int_t i = 0; i < ts_array.extent()[0]; ++i) {
			//通过pointi来访问tensor
			printf("(%d, %d): %f\n", i, j, ts_array(pointi<2>{i, j}));
		}
	}

#ifdef MATAZURE_CUDA

	//构造一个linear access的二维int型lambda_tensor
	auto clts_linear = make_lambda(pointi<2>{ 4, 4 }, [] __matazure__ (int_t i)->int{
		return i;
	});
	auto cts_linear = clts_linear.persist();
	auto ts_linear_cp = tensor<int, 2>(clts_linear.extent());
	mem_copy(cts_linear, ts_linear_cp);
	printf("linear access ts_linear_cp.\n");
	for (int_t i = 0; i < ts_linear_cp.size(); ++i) {
		printf("%d: %d\n", i, ts_linear_cp[i]);
	}

	//构造一个array access的二维float型lambda_tensor
	auto clts_array = make_lambda(pointi<2>{ 4, 4 }, [] __matazure__ (pointi<2> idx)->float{
		return idx[0] + idx[1] * 0.1f;
	});
	tensor<float, 2> ts_array_cp(clts_array.extent());
	mem_copy(clts_array.persist(), ts_array_cp);
	printf("array access ts_array_cp.\n");
	//按数组访问的方式打印值
	for (int_t j = 0; j < ts_array_cp.extent()[1]; ++j) {
		for (int_t i = 0; i < ts_array_cp.extent()[0]; ++i) {
			//通过pointi来访问tensor
			printf("(%d, %d): %f\n", i, j, ts_array_cp(pointi<2>{i, j}));
		}
	}

	cu_tensor<int, 2> tsfdsag;

#endif

	return 0;
}
