#include <matazure/tensor>
#include <chrono>

using namespace matazure;

int main(){
	tensor<float, 2> ts_input(pointi<2>{224, 224});
	tensor<float, 2> ts_output(ts_input.shape());
	//static_tensor<float, dim<112, 112>> ts_input{};
	static_tensor<float, dim<3, 3>> sts_kenel{};
	//tensor<float, 2> sts_kenel(3, 3);
	
	auto t0 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 10000; ++i) {
		conv_direct(ts_input, sts_kenel, ts_output);
	}

	auto t1 = (std::chrono::high_resolution_clock::now() - t0).count();
	printf("cost time: %f ms\n", t1 / 1000000.0);

	return 0;
}
