#include "image_helper.hpp"
#include <matazure/tensor>

using namespace matazure;

int main(int argc, char * argv[]) {
	if (argc < 2) {
		std::cout << "sample6_convolution input_image" << std::endl;
		return -1;
	}

	tensor<pointb<3>, 2> img_rgb = read_rgb_image(argv[1]);
	tensor<pointf<3>, 2> kernel_mean(pointi<2>{3, 3});
	fill(kernel_mean, pointf<3>{0.111f, 0.111f, 0.111f});

	cuda::tensor<pointb<3>, 2> cimg_rgb(img_rgb.shape());
	mem_copy(img_rgb, cimg_rgb);
	cuda::tensor<pointf<3>, 2> ckernel_mean(kernel_mean.shape());
	mem_copy(kernel_mean, ckernel_mean);

	typedef dim<16, 16> BLOCK_SIZE;
	pointi<2> block_size{16, 16};
	auto valid_block_dim = block_size - kernel_mean.shape() + pointi<2>{1, 1};
	auto grid_dim = (img_rgb.shape() + valid_block_dim - pointi<2>{1, 1}) / valid_block_dim;
	auto padding = kernel_mean.shape() / 2;

	//拷贝padding的图像
	cuda::tensor<pointf<3>, 2, padding_layout<2>> cimg_padding(img_rgb.shape(), padding, padding);
	cuda::for_index(cimg_rgb.shape(), [=] __device__ (pointi<2> idx) {
		cimg_padding(idx) = point_cast<float>(cimg_rgb(idx));
	});
	cuda::device_synchronize();

	cuda::tensor<pointf<3>, 2> cimg_mean(img_rgb.shape());

	cuda::block_for_index<BLOCK_SIZE>(grid_dim, [=] __device__ (cuda::block_index<BLOCK_SIZE> block_idx){
		auto valid_global_idx = valid_block_dim * block_idx.block + block_idx.local - padding;
		__shared__ static_tensor<pointf<3>, BLOCK_SIZE> sh_ts_block;

		if (inside_range(valid_global_idx, -padding, cimg_padding.shape() + padding)) {
			sh_ts_block(block_idx.local) = cimg_padding(valid_global_idx);
		}

		cuda::device::barrier();

		if (inside_range(block_idx.local, padding, block_idx.block_dim - padding)
			&& inside_range(valid_global_idx, pointi<2>::zeros(), cimg_padding.shape())) {
			auto sum = zero<pointf<3>>::value();
			cuda::device::for_index(pointi<2>::zeros(), ckernel_mean.shape(), [&](const pointi<2> &idx) {
				sum += sh_ts_block(block_idx.local + idx - padding) * ckernel_mean(idx);
			});
			cimg_mean[valid_global_idx] = sum;
		}
	});
	cuda::device_synchronize();

	cuda::tensor<pointb<3>, 2> cimg_mean_byte(cimg_mean.shape());
	cuda::transform(cimg_mean, cimg_mean_byte, [] __device__ (pointf<3> pixel) {
		return point_cast<byte>(pixel);
	});

	tensor<pointb<3>, 2> img_mean(cimg_mean_byte.shape());
	mem_copy(cimg_mean_byte, img_mean);

	write_rgb_png("mean_" + std::string(argv[1]), img_mean);

	return 0;
}
