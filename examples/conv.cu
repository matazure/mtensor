#include <matazure/tensor>
#include <matazure/cuda/puzzle/conv.hpp>

using namespace matazure;

__constant__ static_tensor<float,dim< 3,  3>> mask;
MATAZURE_PUZZEL_CONV_GLOBAL(conv_global, mask)
MATAZURE_PUZZEL_CONV_BLOCK(conv_block, mask)
MATAZURE_PUZZEL_CONV_BLOCK_WITH_CRACK(conv_block_crack, mask)

int main() {
	try {
		static_tensor<float,dim< 3,  3>> host_mask;
		fill(host_mask, 1.0f / host_mask.size());
		cuda::copy_symbol(host_mask, mask);

		tensor<byte, 2> gray(512, 512);
		io::read_raw_data("data/lena_gray8_512x512.raw_data", gray);
		auto cu_gray = mem_clone(gray, device_t{});

		auto lcts_conv = cuda::puzzle::conv_global(tensor_cast<float>(clamp_zero(cu_gray)));
		auto cts_conv = apply(lcts_conv, op::saturate_convert<byte>{}).persist();
		cuda::device_synchronize();
		auto ts_conv = mem_clone(cts_conv, host_t{});
		io::write_raw_data("data/lena_gray8_conv_512x512.raw_data", ts_conv);

		auto cts_conv_block = cuda::puzzle::conv_block<dim<16, 16>>(tensor_cast<float>(cu_gray));
		auto cts_byte_conv_block = apply(cts_conv_block, op::saturate_convert<byte>{}).persist();
		cuda::device_synchronize();
		auto ts_byte_conv_block = mem_clone(cts_byte_conv_block, host_t{});
		io::write_raw_data("data/lena_gray8_conv_block_512x512.raw_data", ts_byte_conv_block);

		auto cts_conv_block_crack = cuda::puzzle::conv_block_crack<dim<32, 32>>(tensor_cast<float>(clamp_zero(cu_gray)));
		auto cts_byte_conv_block_crack = apply(cts_conv_block_crack, op::saturate_convert<byte>{}).persist();
		cuda::device_synchronize();
		auto ts_byte_conv_block_crack = mem_clone(cts_byte_conv_block_crack, host_t{});
		io::write_raw_data("data/lena_gray8_conv_block_crack_512x512.raw_data", ts_byte_conv_block_crack);
	}
	catch (std::exception &e) {
		std::cout << "unexpected error: " << e.what() << std::endl;
	}
}
