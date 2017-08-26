#include <matazure/tensor>
#include <image_utility.hpp>

using namespace matazure;

//nvcc bug walkaround
using cu_rgb_float_image = cuda::tensor<pointf<3>, 2>;
using cu_rgb_byte_image = cuda::tensor<pointb<3>, 2>;

__constant__ static_tensor<pointf<3>, dim< 3, 3>> mask;
//之所以使用宏来定义卷积函数，是因为cuda的constant内存必须全局申明
//声明一个叫做conv_global的卷积函数 用mask作为卷积核
MATAZURE_CUDA_PUZZEL_CONV_GLOBAL(conv_global, mask)
//分块的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK(conv_block, mask)
//边缘不处理的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_CRACK(conv_block_crack, mask)
//处理了边缘的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_OVERLAP(conv_block_overlap, mask)

int main(int argc, char *argv[]) {
	if (argc < 2){
		printf("please input a 3 channel(rbg) image path");
		return -1;
	}
	auto ts_rgb = read_rgb_image(argv[1]);

	static_tensor<pointf<3>, dim< 3, 3>> host_mask;
	//使用均值卷积核
	fill(host_mask, pointf<3>::all(1.0f) / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	auto pointf3_to_pointb3 = [] __matazure__ (pointf<3> x){
		pointb<3> re{};
		auto convertor = unary::saturate_convertor<byte>{};
		re[0] = convertor(x[0]);
		re[1] = convertor(x[1]);
		re[2] = convertor(x[2]);
		return re;
	};

	auto cts_rgb = mem_clone(ts_rgb, device_tag{});
	auto lcts_conv = cuda::puzzle::conv_global(tensor_cast<pointf<3>>(clamp_zero(cts_rgb)));
	auto cts_conv = apply(lcts_conv, pointf3_to_pointb3).persist();
	cuda::device_synchronize();
	auto ts_conv = mem_clone(cts_conv, host_tag{});
	auto conv_global_image_path = argc > 2 ? argv[2] : "conv_global.png";
	write_rgb_png( conv_global_image_path, ts_conv);

	cuda::tensor<pointf<3>, 2> cts_conv_block(cts_rgb.shape());
	cuda::puzzle::conv_block<dim<16, 16>>(tensor_cast<pointf<3>>(cts_rgb), cts_conv_block);
	auto cts_pointb3_conv_block = apply(cts_conv_block, pointf3_to_pointb3).persist();
	cuda::device_synchronize();
	auto ts_conv_block = mem_clone(cts_pointb3_conv_block, host_tag{});
	auto conv_block_image_path = argc > 3 ? argv[3] : "conv_block.png";
	write_rgb_png(conv_block_image_path, ts_conv_block);

	cuda::tensor<pointf<3>, 2> cts_conv_block_crack(cts_rgb.shape());
	cuda::puzzle::conv_block_crack<dim<32, 32>>(tensor_cast<pointf<3>>(clamp_zero(cts_rgb)), cts_conv_block_crack);
	auto cts_pointb3_conv_block_crack = apply(cts_conv_block_crack, pointf3_to_pointb3).persist();
	cuda::device_synchronize();
	auto ts_conv_block_crack = mem_clone(cts_pointb3_conv_block_crack, host_tag{});
	auto conv_block_crack_image_path = argc > 4 ? argv[4] : "conv_block_crack.png";
	write_rgb_png(conv_block_crack_image_path, ts_conv_block_crack);

	cuda::tensor<pointf<3>, 2> cts_conv_block_overlap(cts_rgb.shape());
	cuda::puzzle::conv_block_overlap<dim<16, 16>>(tensor_cast<pointf<3>>(clamp_zero(cts_rgb)), cts_conv_block_overlap);
	auto cts_pointb3_conv_block_overlap = apply(cts_conv_block_overlap,pointf3_to_pointb3).persist();
	cuda::device_synchronize();
	auto ts_conv_block_overlap = mem_clone(cts_pointb3_conv_block_overlap, host_tag{});
	auto conv_block_overlap_image_path = argc > 5 ? argv[5] : "conv_block_overlap.png";
	write_rgb_png(conv_block_overlap_image_path, ts_conv_block_overlap);

	return 0;
}
