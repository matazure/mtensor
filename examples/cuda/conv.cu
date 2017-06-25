#include <matazure/tensor>

using namespace matazure;

__constant__ static_tensor<float, dim< 3, 3>> mask;
//之所以使用宏来定义卷积函数，是因为cuda的constant内存必须全局申明
//声明一个叫做conv_global的卷积函数 用mask作为卷积核
MATAZURE_CUDA_PUZZEL_CONV_GLOBAL(conv_global, mask)
//分块的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_ALIGNED(conv_block, mask)
//边缘不处理的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_CRACK(conv_block_crack, mask)
//处理了边缘的卷积实现
MATAZURE_CUDA_PUZZEL_CONV_BLOCK_OVERLAP(conv_block_overlap, mask)

int main() {
	static_tensor<float, dim< 3, 3>> host_mask;
	fill(host_mask, 1.0f / host_mask.size());
	cuda::copy_symbol(host_mask, mask);

	tensor<byte, 2> gray(512, 512);
	io::read_raw_data("data/lena_gray8_512x512.raw_data", gray);
	auto cu_gray = mem_clone(gray, device_t{});

	auto lcts_conv = cuda::puzzle::conv_global(tensor_cast<float>(clamp_zero(cu_gray)));
	auto cts_conv = apply(lcts_conv, unary::saturate_convert<byte>{}).persist();
	cuda::device_synchronize();
	auto ts_conv = mem_clone(cts_conv, host_t{});
	io::write_raw_data("data/lena_gray8_conv_512x512.raw_data", ts_conv);

	cuda::tensor<float, 2> cts_conv_block(cu_gray.shape());
	cuda::puzzle::conv_block<dim<16, 16>>(tensor_cast<float>(cu_gray), cts_conv_block);
	auto cts_byte_conv_block = apply(cts_conv_block, unary::saturate_convert<byte>{}).persist();
	cuda::device_synchronize();
	auto ts_byte_conv_block = mem_clone(cts_byte_conv_block, host_t{});
	io::write_raw_data("data/lena_gray8_conv_block_512x512.raw_data", ts_byte_conv_block);

	cuda::tensor<float, 2> cts_conv_block_crack(cu_gray.shape());
	cuda::puzzle::conv_block_crack<dim<32, 32>>(tensor_cast<float>(clamp_zero(cu_gray)), cts_conv_block_crack);
	auto cts_byte_conv_block_crack = apply(cts_conv_block_crack, unary::saturate_convert<byte>{}).persist();
	cuda::device_synchronize();
	auto ts_byte_conv_block_crack = mem_clone(cts_byte_conv_block_crack, host_t{});
	io::write_raw_data("data/lena_gray8_conv_block_crack_512x512.raw_data", ts_byte_conv_block_crack);

	cuda::tensor<float, 2> cts_conv_block_overlap(cu_gray.shape());
	cuda::puzzle::conv_block_overlap<dim<16, 16>>(tensor_cast<float>(clamp_zero(cu_gray)), cts_conv_block_overlap);
	auto cts_byte_conv_block_overlap = apply(cts_conv_block_overlap, unary::saturate_convert<byte>{}).persist();
	cuda::device_synchronize();
	auto ts_byte_conv_block_overlap = mem_clone(cts_byte_conv_block_overlap, host_t{});
	io::write_raw_data("data/lena_gray8_conv_block_overlap_512x512.raw_data", ts_byte_conv_block_overlap);

	return 0;
}
