#include <matazure/tensor>

using namespace matazure;

int main(int argc, char *argv[]) {
	tensor<point<byte, 3>, 2> ts_rgb(512, 512);
	io::read_raw_data("data/lena_rgb888_512x512.raw_data", ts_rgb);

#ifdef MATAZURE_CUDA
	auto cts_rgb = mem_clone(ts_rgb, device_t{});
	auto lcts_rgb_shift_zero = cts_rgb - point<byte, 3>{128, 128, 128};
	auto lcts_rgb_stride = stride(lcts_rgb_shift_zero, 2);
	auto lcts_rgb_normalized = tensor_cast<pointf<3>>(lcts_rgb_stride) / pointf<3>{128.0f, 128.0f, 128.0f};
	auto cts_rgb_normalized = lcts_rgb_normalized.persist();
	auto ts_rgb_normalized = mem_clone(cts_rgb_normalized, host_t{});
#else
	auto lts_rgb_shift_zero = ts_rgb - point<byte, 3>{128, 128, 128};
	auto lts_rgb_stride = stride(lts_rgb_shift_zero, 2);
	auto lts_rgb_normalized = tensor_cast<pointf<3>>(lts_rgb_stride) / pointf<3>{128.0f, 128.0f, 128.0f};
	auto ts_rgb_normalized = lts_rgb_normalized.persist();
#endif

	tensor<float, 2> ts_red(ts_rgb_normalized.shape());
	tensor<float, 2> ts_green(ts_rgb_normalized.shape());
	tensor<float, 2> ts_blue(ts_rgb_normalized.shape());
	auto ts_zip_point = point_view(zip(ts_red, ts_green, ts_blue));
	copy(ts_rgb_normalized, ts_zip_point);

	io::write_raw_data("data/lena_red_float_256x256.raw_data", ts_red);
	io::write_raw_data("data/lena_green_float_256x256.raw_data", ts_green);
	io::write_raw_data("data/lena_blue_float_256x256.raw_data", ts_blue);

	return 0;
}
