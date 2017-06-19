#include <matazure/tensor>

using namespace matazure;

int main() {
	cuda::matrix<float, row_major_t> cmat_lhs(17, 38);
	fill(cmat_lhs, 1.0f);
	cuda::matrix<float> cmat_rhs(38, 19);
	fill(cmat_rhs, 2.0f);

	auto cmat_re = numeric::prod_general(cmat_lhs, cmat_rhs).persist();
	cuda::device_synchronize();

	auto cmat_re_block = cuda::numeric::prod_block<16>(cmat_lhs, cmat_rhs);
	cuda::device_synchronize();
	
	auto mat_re = mem_clone(cmat_re, host_t{});
	auto mat_re_block = mem_clone(cmat_re_block, host_t{});

	for (int_t j = 0; j < mat_re.shape()[1]; ++j) {
		for (int_t i = 0; i < mat_re.shape()[0]; ++i) {
			MATAZURE_ASSERT(mat_re(i, j) == mat_re_block(i, j), "failed");
		}
	}

	return 0;
}
