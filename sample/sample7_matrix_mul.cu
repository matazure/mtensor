#include <matazure/tensor>

using namespace matazure;

int main(int argc, char * argv[]) {
	const int BLOCK_SIZE = 16;
	typedef dim<BLOCK_SIZE, BLOCK_SIZE> BLOCK_DIM;
	pointi<2> block_dim = BLOCK_DIM::value();
	pointi<2> grid_dim{8, 8};
	pointi<2> global_dim = block_dim * grid_dim;
	int M = global_dim[0];
	int N = global_dim[1];
	int K = BLOCK_SIZE * 4;

	cuda::tensor<float, 2> cmat_a(pointi<2>{M, K});
	cuda::tensor<float, 2> cmat_b(pointi<2>{K, N});
	cuda::tensor<float, 2> cmat_c(pointi<2>{M, N});

	cuda::fill(cmat_a, 1.0f);
	cuda::fill(cmat_b, 2.0f);
	cuda::device_synchronize();

	cuda::block_for_index<BLOCK_DIM>(grid_dim, [=] __device__ (cuda::block_index<BLOCK_DIM> block_idx) {
		auto row = block_idx.local[0];
		auto col = block_idx.local[1];
		auto global_row = block_idx.global[0];
		auto global_col = block_idx.global[1];

		__shared__ static_tensor<float, BLOCK_DIM> local_a;
		__shared__ static_tensor<float, BLOCK_DIM> local_b;

		float sum = 0.0f;
		for (int_t i = 0; i < K; i += BLOCK_SIZE) {
			local_a(row, col) = cmat_a(global_row, col + i);
			local_b(row, col) = cmat_b(row + i, global_col);
			cuda::sync_threads();

			for (int_t N = 0; N < BLOCK_SIZE; N++) {
				sum += local_a(row, N) * local_b(N, col);
			}
		}

		cmat_c(block_idx.global) = sum;
	});
	cuda::device_synchronize();

	return 0;
}
