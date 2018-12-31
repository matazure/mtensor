///tensor自带的多个不同版本的卷积运算， 适用于不同类型的，不同维度（最大四维， 可以自己拓展)的卷积运算

#include <matazure/tensor>
#include <image_utility.hpp>

using namespace matazure;

//nvcc bug walkaround
using cu_rgb_float_image = cuda::tensor<pointf<3>, 2>;
using cu_rgb_byte_image = cuda::tensor<pointb<3>, 2>;

__constant__ static_tensor<pointf<3>, dim< 3, 3>> constant_ts_kernel;
//之所以使用宏来定义卷积函数，是因为cuda的constant内存必须全局申明
//声明一个叫做conv_lazy_array_index_unclamp_constant_kernel的卷积函数 用constant_ts_kernel作为卷积核
MATAZURE_CUDA_PUZZEL_CONV_LAZY_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_lazy_array_index_unclamp_constant_kernel, constant_ts_kernel)
//分块的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_ARRAY_INDEX_UNCLAME_CONSTANT_KERNEL(conv_block_array_index_unclamp_constant_kernel, constant_ts_kernel)
//边缘不处理的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_CRACK_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_block_crack_array_index_unclamp_constant_kernel, constant_ts_kernel)
//处理了边缘的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_OVERLAP_ARRAY_INDEX_UNCLAMP_CONSTANT_KERNEL(conv_block_overlap_array_index_unclamp_constant_kernel, constant_ts_kernel)

int main(int argc, char *argv[]) {
	if (argc < 2){
		printf("please input a 3 channel(rbg) image path");
		return -1;
	}
	auto ts_rgb = read_rgb_image(argv[1]);

	static_tensor<pointf<3>, dim< 3, 3>> sts_kernel;
	//使用均值卷积核
	fill(sts_kernel, pointf<3>::all(1.0f) / sts_kernel.size());
	cuda::copy_symbol(sts_kernel, constant_ts_kernel);

	typedef point<byte, 3> (* sature_cast_op)(const point<float, 3> &);
	sature_cast_op pointf3_to_pointb3 = &unary::saturate_cast<byte, float, 3>;

	auto cts_rgb = mem_clone(ts_rgb, device_tag{});

	{
		auto lcts_conv = cuda::puzzle::conv_lazy_array_index_unclamp_constant_kernel(cast<pointf<3>>(clamp_zero(cts_rgb)));
		auto cts_conv = apply(lcts_conv, pointf3_to_pointb3).persist();
		cuda::device_synchronize();
		auto ts_conv = mem_clone(cts_conv, host_tag{});
		auto conv_global_image_path = argc > 2 ? argv[2] : "conv_lazy_array_index_unclamp_constant_kernel.png";
		write_rgb_png( conv_global_image_path, ts_conv);
	}

	{
		cuda::tensor<pointf<3>, 2> cts_conv_block(cts_rgb.shape());
		cuda::puzzle::conv_block_array_index_unclamp_constant_kernel<dim<16, 16>>(cast<pointf<3>>(clamp_zero(cts_rgb)), cts_conv_block);
		auto cts_pointb3_conv_block = apply(cts_conv_block, pointf3_to_pointb3).persist();
		cuda::device_synchronize();
		auto ts_conv_block = mem_clone(cts_pointb3_conv_block, host_tag{});
		auto conv_block_image_path = argc > 3 ? argv[3] : "conv_block_array_index_unclamp_constant_kernel.png";
		write_rgb_png(conv_block_image_path, ts_conv_block);
	}

	{
		cuda::tensor<pointf<3>, 2> cts_conv_block_crack(cts_rgb.shape());
		cuda::puzzle::conv_block_crack_array_index_unclamp_constant_kernel<dim<32, 32>>(cast<pointf<3>>(clamp_zero(cts_rgb)), cts_conv_block_crack);
		auto cts_pointb3_conv_block_crack = apply(cts_conv_block_crack, pointf3_to_pointb3).persist();
		cuda::device_synchronize();
		auto ts_conv_block_crack = mem_clone(cts_pointb3_conv_block_crack, host_tag{});
		auto conv_block_crack_image_path = argc > 4 ? argv[4] : "conv_block_crack_array_index_unclamp_constant_kernel.png";
		write_rgb_png(conv_block_crack_image_path, ts_conv_block_crack);
	}

	{
		cuda::tensor<pointf<3>, 2> cts_conv_block_overlap(cts_rgb.shape());
		cuda::puzzle::conv_block_overlap_array_index_unclamp_constant_kernel<dim<16, 16>>(cast<pointf<3>>(clamp_zero(cts_rgb)), cts_conv_block_overlap);
		auto cts_pointb3_conv_block_overlap = apply(cts_conv_block_overlap, pointf3_to_pointb3).persist();
		cuda::device_synchronize();
		auto ts_conv_block_overlap = mem_clone(cts_pointb3_conv_block_overlap, host_tag{});
		auto conv_block_overlap_image_path = argc > 5 ? argv[5] : "conv_block_overlap_array_index_unclamp_constant_kernel.png";
		write_rgb_png(conv_block_overlap_image_path, ts_conv_block_overlap);
	}

	{
		auto lcts_conv = puzzle::conv_lazy_array_index_unclamp(cast<pointf<3>>(clamp_zero(cts_rgb)), sts_kernel);
		auto cts_conv = apply(lcts_conv, pointf3_to_pointb3).persist();
		cuda::device_synchronize();
		auto ts_conv = mem_clone(cts_conv, host_tag{});
		auto conv_global_image_path = argc > 6 ? argv[6] : "conv_lazy_array_index_unclamp.png";
		write_rgb_png( conv_global_image_path, ts_conv);
	}

	return 0;
}
