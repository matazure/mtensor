#include <mtensor.hpp>

using namespace matazure;

int main(int argc, char* argv[]) {
    const int BLOCK_SIZE = 16;                      // block尺寸位16x16
    typedef dim<BLOCK_SIZE, BLOCK_SIZE> BLOCK_DIM;  // 需要用一个dim<16, 16>来表示编译时block尺寸
    point2i block_dim = BLOCK_DIM::value();  //将编译时的block尺寸转换为运行时point2i类型
    point2i grid_dim{8, 8};                  // grid的尺寸，决定block的数目，布局
    point2i global_dim = block_dim * grid_dim;  // 全局尺寸

    int M = global_dim[0];
    int N = global_dim[1];
    int K = BLOCK_SIZE * 4;
    cuda::tensor<float, 2> cmat_a(point2i{M, K});
    cuda::tensor<float, 2> cmat_b(point2i{K, N});
    cuda::tensor<float, 2> cmat_c(point2i{M, N});
    // block_for_index需要给一个编译时的block尺寸， grid_dim是运行时的grid尺寸
    cuda::block_for_index<BLOCK_DIM>(grid_dim,
                                     [=] __device__(cuda::block_index<BLOCK_DIM> block_idx) {
                                         auto row = block_idx.local[0];
                                         auto col = block_idx.local[1];
                                         auto global_row = block_idx.global[0];
                                         auto global_col = block_idx.global[1];
                                         //位于shared内存的分块矩阵
                                         __shared__ local_tensor<float, BLOCK_DIM> local_a;
                                         __shared__ local_tensor<float, BLOCK_DIM> local_b;

                                         float sum = 0.0f;
                                         for (int_t i = 0; i < K; i += BLOCK_SIZE) {
                                             //拷贝局部矩阵块
                                             local_a(row, col) = cmat_a(global_row, col + i);
                                             local_b(row, col) = cmat_b(row + i, global_col);
                                             cuda::syncthreads();
                                             //矩阵块乘法
                                             for (int_t N = 0; N < BLOCK_SIZE; N++) {
                                                 sum += local_a(row, N) * local_b(N, col);
                                             }
                                             cuda::syncthreads();
                                         }
                                         cmat_c(block_idx.global) = sum;
                                     });

    return 0;
}
