#include <matazure/tensor>
#include <chrono>

using namespace matazure;

int main(){

	tensor<float, 2> ts_input(224, 224);
	tensor<float, 2> ts_output(ts_input.shape());
	//static_tensor<float, dim<112, 112>> ts_input{};
	static_tensor<float, dim<1, 1>> sts_kenel{};
	//tensor<float, 2> sts_kenel(3, 3);


	auto t0 = std::chrono::high_resolution_clock::now();

	//48ms  caffe2
	for (int i = 0; i < 10000; ++i) {
		auto tmp = puzzle::conv_general(ts_input, sts_kenel).persist();
		//ts_input[0] = 3.0;
		// puzzle::conv_direct(ts_input, sts_kenel, ts_output);
		//ts_input[100] = 4.0;
		//auto tmp = (ts_input * 2.0f).persist();
		//auto tmp = make_lambda(ts_input.shape(), [=](int_t i) { return ts_input[i] * 2.0f; }).persist();
	}

	auto t1 = (std::chrono::high_resolution_clock::now() - t0).count();
	printf("cost time: %f ms\n", t1 / 1000000.0);

	return int(ts_output[0]);
}
