#include <matazure/tensor>

using namespace matazure;

int main(){
	cuda::matrix<float, row_major_t> cmat_lhs(10, 100);
	fill(cmat_lhs, 1.0f);
	cuda::matrix<float> cmat_rhs(100, 10);
	fill(cmat_rhs, 2.0f);

	auto cmat_re = puzzle::prod_general(cmat_lhs, cmat_rhs).persist();
	cuda::device_synchronize();

	auto mat_re = mem_clone(cmat_re, host_t{});

	return 0;
}
