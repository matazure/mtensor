#include "../bm_config.hpp"

// copy from cuda-samples
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void bm_cuda_sample_matrix_mul(benchmark::State& state) {
    const int BLOCK_SIZE = 32;
    typedef dim<BLOCK_SIZE, BLOCK_SIZE> BLOCK_DIM;
    pointi<2> block_dim = BLOCK_DIM::value();
    pointi<2> grid_dim;
    fill(grid_dim, state.range(0));
    pointi<2> global_dim = block_dim * grid_dim;

    int M = global_dim[0];
    int N = global_dim[1];
    int K = BLOCK_SIZE * 100;

    cuda::tensor<float, 2> cmat_a(pointi<2>{M, K});
    cuda::tensor<float, 2> cmat_b(pointi<2>{K, N});
    cuda::tensor<float, 2> cmat_c(pointi<2>{M, N});

    while (state.KeepRunning()) {
        // cudaEvent_t events[2];
        // cudaEventCreate(&events[0]);
        // cudaEventCreate(&events[1]);

        // cudaEventRecord(events[0], nullptr);

        MatrixMulCUDA<BLOCK_SIZE><<<cuda::internal::pointi_to_dim3(grid_dim),
                                    cuda::internal::pointi_to_dim3(block_dim)>>>(
            cmat_c.data(), cmat_b.data(), cmat_a.data(), cmat_b.shape()[0], cmat_a.shape()[0]);

        // cudaEventRecord(events[1], nullptr);
        // cudaEventSynchronize(events[1]);

        // float avg_ms;
        // cudaEventElapsedTime(&avg_ms, events[0], events[1]);

        // state.SetIterationTime(avg_ms / 1000.0f);

        // cudaEventDestroy(events[0]);
        // cudaEventDestroy(events[1]);
        cudaStreamSynchronize(nullptr);

        cuda::assert_runtime_success(cudaGetLastError());

        benchmark::DoNotOptimize(cmat_c.data());
    }

    state.SetItemsProcessed(state.iterations() * M * N * K);
}

BENCHMARK(bm_cuda_sample_matrix_mul)->Arg(64);

void bm_cuda_matrix_mul(benchmark::State& state) {
    const int BLOCK_SIZE = 32;
    typedef dim<BLOCK_SIZE, BLOCK_SIZE> BLOCK_DIM;
    pointi<2> block_dim = BLOCK_DIM::value();
    pointi<2> grid_dim;
    fill(grid_dim, state.range(0));
    pointi<2> global_dim = block_dim * grid_dim;

    int M = global_dim[0];
    int N = global_dim[1];
    int K = BLOCK_SIZE * 100;

    cuda::tensor<float, 2> cmat_a(pointi<2>{M, K});
    cuda::tensor<float, 2> cmat_b(pointi<2>{K, N});
    cuda::tensor<float, 2> cmat_c(pointi<2>{M, N});

    while (state.KeepRunning()) {
        cuda::block_for_index<BLOCK_DIM>(grid_dim,
                                         [=] __device__(cuda::block_index<BLOCK_DIM> block_idx) {
                                             auto row = block_idx.local[0];
                                             auto col = block_idx.local[1];
                                             auto global_row = block_idx.global[0];
                                             auto global_col = block_idx.global[1];

                                             float sum = 0.0f;
                                             for (int_t i = 0; i < K; i += BLOCK_SIZE) {
                                                 __shared__ local_tensor<float, BLOCK_DIM> local_a;
                                                 __shared__ local_tensor<float, BLOCK_DIM> local_b;
                                                 local_a(row, col) = cmat_a(global_row, col + i);
                                                 local_b(row, col) = cmat_b(row + i, global_col);
                                                 cuda::syncthreads();

#pragma unroll
                                                 for (int_t N = 0; N < BLOCK_SIZE; N++) {
                                                     sum += local_a(row, N) * local_b(N, col);
                                                 }

                                                 cuda::syncthreads();
                                             }

                                             cmat_c(block_idx.global) = sum;
                                         });
    }

    state.SetItemsProcessed(state.iterations() * M * N * K);
}

BENCHMARK(bm_cuda_matrix_mul)->Arg(64);