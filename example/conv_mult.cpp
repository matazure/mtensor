#include <matazure/tensor>
#include <image_utility.hpp>

using namespace matazure;

int main(int argc, char *argv[]) {
	if (argc < 2){
		printf("please input a 3 channel(rbg) image path");
		return -1;
	}
	auto ts_rgb = read_rgb_image(argv[1]);

	static_tensor<pointf<3>, dim< 3, 3>> sts_kernel;

	fill(sts_kernel, pointf<3>::all(1.0f) / static_cast<float>(sts_kernel.size()));

	typedef point<byte, 3> (* sature_cast_op)(const point<float, 3> &);
	sature_cast_op pointf3_to_pointb3 = &unary::saturate_cast<byte, float, 3>;

	{
		auto lts_conv = puzzle::conv_lazy_array_index_inside_clamp_zero(cast<pointf<3>>(ts_rgb), sts_kernel);
	#ifdef MATAZURE_OPENMP
		omp_policy policy{};
	#else
		sequence_policy policy{};
	#endif
		auto ts_conv = apply(lts_conv, pointf3_to_pointb3).persist(policy);

		auto image_path = argc > 2 ? argv[2] : "conv_lazy_array_index_inside_clamp_zero.png";
		write_rgb_png(image_path, ts_conv);
	}

	return 0;
}
