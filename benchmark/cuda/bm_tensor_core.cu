

#include <mma.h>
#include "../bm_config.hpp"

using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// // Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
// //  1) Matrices are packed in memory.
// //  2) M, N and K are multiples of 16.
// //  3) Neither A nor B are transposed.
// // Note: This is NOT a high performance example but is for demonstration purposes only
// //       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half* a, half* b, float* c, int M, int N, int K) {
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    wmma::store_matrix_sync(c + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);
}

void bm_cuda_wmma_example(benchmark::State& state) {
    // Must be multiples of 16 for wmma code to work
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    typedef dim<128, 4> BLOCK_DIM;
    pointi<2> block_dim = BLOCK_DIM::value();

    pointi<2> warp_dim = {16, 16};

    cuda::tensor<half, 2> cmat_a(pointi<2>{M, K});
    cuda::tensor<half, 2> cmat_b(pointi<2>{K, N});
    cuda::tensor<float, 2> cmat_c(pointi<2>{M, N});

    auto grid_dim = cmat_c.shape() / pointi<2>{64, 64};

    while (state.KeepRunning()) {
        wmma_example<<<cuda::internal::pointi_to_dim3(grid_dim), dim3(128, 4)>>>(
            cmat_a.data(), cmat_b.data(), cmat_c.data(), M, N, K);

        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(state.iterations() * M * N * K);
}

BENCHMARK(bm_cuda_wmma_example);

void bm_cuda_wmma(benchmark::State& state) {
    // Must be multiples of 16 for wmma code to work
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    typedef dim<128, 4> BLOCK_DIM;
    pointi<2> block_dim = BLOCK_DIM::value();

    pointi<2> warp_dim = {16, 16};

    cuda::tensor<half, 2> cmat_a(pointi<2>{M, K});
    cuda::tensor<half, 2> cmat_b(pointi<2>{K, N});
    cuda::tensor<float, 2> cmat_c(pointi<2>{M, N});

    auto grid_dim = cmat_c.shape() / pointi<2>{64, 64};

    while (state.KeepRunning()) {
        cuda::block_for_index<BLOCK_DIM>(grid_dim, [=] __device__(
                                                       cuda::block_index<BLOCK_DIM> block_idx) {
            int warpM = block_idx.global[0] / 32;
            int warpN = block_idx.global[1];

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

            // Loop over k
            for (int i = 0; i < K; i += WMMA_K) {
                int aRow = warpM * WMMA_M;
                int aCol = i;

                int bRow = i;
                int bCol = warpN * WMMA_N;

                // Bounds checking
                if (aRow < M && aCol < K && bRow < K && bCol < N) {
                    // Load the inputs
                    wmma::load_matrix_sync(a_frag, &cmat_a(aRow, aCol), cmat_a.shape()[0]);
                    wmma::load_matrix_sync(b_frag, &cmat_b(bRow, bCol), cmat_b.shape()[0]);

                    // Perform the matrix multiplication
                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
            }

            auto cRow = warpM * WMMA_M;
            auto cCol = warpN * WMMA_N;

            wmma::store_matrix_sync(&cmat_c(cRow, cCol), acc_frag, cmat_c.shape()[0],
                                    wmma::mem_col_major);
        });
    }

    state.SetItemsProcessed(state.iterations() * M * N * K);
}

BENCHMARK(bm_cuda_wmma);