#ifndef __CUDACC__

#include <matazure/tensor>
#include <chrono>

using namespace matazure;
using io::read_raw_data;
using io::write_raw_data;
using namespace std::chrono;

int main(int argc, char *argv[]) {
	std::string input_raw_data_path = "data/lena_rgb888_512x512.raw_data";

	using rgb_t = point<byte, 3>;
	tensor<rgb_t, 2> ts_rgb(512, 512);
	read_raw_data(input_raw_data_path, ts_rgb);

	auto lts_red = make_lambda(ts_rgb.extent(), [=](int_t i) {
		return ts_rgb[i][0];
	}).persist();

	auto t0 = high_resolution_clock::now();



	/*auto ts_normalized_red = make_lambda(lts_red.extent(), [=](int_t i) {
		return lts_red[i] * 100;
	}).persist();*/

	//auto ts_normalized_red = make_lambda(lts_red.extent(), __mul_linear_access_tensor_with_value__<decltype(lts_red)>(lts_red, 100)).persist();


	auto ts_normalized_red = (lts_red / 100).persist();

	//auto ts_normalized_red = make_lambda(ts_rgb.extent(), [=](int_t i) {
	//	return (ts_rgb[i][0] - 256) / 100;
	//}).persist();

	auto t1 = high_resolution_clock::now();

	printf("cost time: %dns\n", (t1 - t0).count());

	std::string output_red_chanel_raw_data_path = "data/lena_red8_256x256.raw_data";
	std::string output_green_chanel_raw_data_path = "data/lena_green8_256x256.raw_data";
	std::string output_blue_chanel_raw_data_path = "data/lena_blue8_256x256.raw_data";

	return 0;
}

#else

#include <matazure/tensor>
#include <chrono>

using namespace matazure;
using io::read_raw_data;
using io::write_raw_data;
using namespace std::chrono;

int main(int argc, char *argv[]) {
	std::string input_raw_data_path = "data/lena_rgb888_512x512.raw_data";

	using rgb_t = point<byte, 3>;
	tensor<rgb_t, 2> ts_rgb(512, 512);
	read_raw_data(input_raw_data_path, ts_rgb);
	auto cts_rgb = mem_clone(ts_rgb, device_t{});

	auto lcts_red = make_lambda(cts_rgb.extent(), [=] __matazure__(int_t i) {
		return cts_rgb[i][0];
	}).persist();

	auto t0 = high_resolution_clock::now();




	auto cts_normalized_red = make_lambda(lcts_red.extent(), [=] __matazure__(int_t i) {
		return lcts_red[i] * 100;
	}).persist();

	//auto ts_normalized_red = make_lambda(lts_red.extent(), __div_linear_access_tensor_with_value__<decltype(lts_red)>(lts_red, 100)).persist();


	//auto cts_normalized_red = (lcts_red / 100).persist();

	//auto ts_normalized_red = make_lambda(ts_rgb.extent(), [=](int_t i) {
	//	return (ts_rgb[i][0] - 256) / 100;
	//}).persist();

	auto t1 = high_resolution_clock::now();

	printf("cost time: %dns\n", (t1 - t0).count());

	std::string output_red_chanel_raw_data_path = "data/lena_red8_256x256.raw_data";
	std::string output_green_chanel_raw_data_path = "data/lena_green8_256x256.raw_data";
	std::string output_blue_chanel_raw_data_path = "data/lena_blue8_256x256.raw_data";

	return 0;
}

#endif
