#pragma hd_warning_disable

#include <matazure/tensor>

using namespace matazure;

int main() {

	tensor<int, 2> ts1(10, 10);
	fill(ts1, 1);
	tensor<int, 2> ts2(ts1.extent());
	fill(ts2, 2);
	tensor<float, 2> ts2f(ts1.extent());
	fill(ts2f, 2.0f);
	tensor<int, 2> ts3(ts1.extent());
	fill(ts3, 3);

	// 返回一个相加的lambda_tensor
	auto lts_1add2 = ts1 + ts2;
	for (int_t i = 0; i < lts_1add2.size(); ++i) {
		printf("%d, ", lts_1add2[i]);
	}
	printf("\n=====================\n");

	auto ts_1add2sub3mul2 = (lts_1add2 - ts3*ts2).persist();

	for (int_t i = 0; i < ts_1add2sub3mul2.size(); ++i) {
		printf("%d, ", ts_1add2sub3mul2[i]);
	}
	printf("\n=====================\n");

	//auto lts_2add2f = ts2 + ts2f; //Compiler error! 类型不匹配
	auto lts_2add2f = tensor_cast<float>(ts2) + ts2f; //Compiler OK

	auto ts_zip = zip(ts1, ts2);
	auto v12 = ts_zip[0];
	printf("ts_zip[0]: (%d, %d)\n", std::get<0>(v12), std::get<1>(v12));

	get<0>(v12) = 14;
	get<1>(v12) = 17;
	printf("ts1[0]: %d; ts2[0]: %d\n", ts1[0], ts2[0]);

#ifdef MATAZURE_CUDA

	cu_tensor<float, 2> cts1(10, 10);
	fill(cts1, 1.0f);
	cu_tensor<float, 2> cts2(10, 10);
	fill(cts2, 2.0f);
	auto lcts_1add2 = cts1 + cts2;

	auto lcts_1add2_by_lambda = make_lambda(cts1.extent(), [=] __matazure__(int_t i) {
		return cts1[i] + cts2[i];
	});

	auto cts_result = (lcts_1add2 - lcts_1add2_by_lambda).persist();
	for_each(cts_result, [] __matazure__ (float e) {
		printf("%.1f, ", e);
	});

#endif

	return 0;
}
