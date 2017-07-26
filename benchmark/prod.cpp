#include <benchmark/benchmark.h>
#include <bm_config.hpp>

#include <matazure/tensor>
#include <emmintrin.h>
#include <immintrin.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdlib>

#ifndef EIGEN_VECTORIZE_SSE2
#error not defined sse2
#endif

using namespace matazure;

void bm_prod_simple_gold(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];
	 //float *p_lhs = (float *)_aligned_malloc(K * M * 4, 16);
	 //float *p_rhs = (float *)_aligned_malloc(K * N * 4, 16);
	 //float *p_re = (float *)_aligned_malloc(M * N * 4, 16);

	while (state.KeepRunning()){
		for (int_t n = 0; n < N; ++n){
			for (int_t m = 0; m < M; ++m){
				float re = 0.0f;
				for (int_t k = 0; k < K; ++k){
					re += p_lhs[k + m * K] * p_rhs[k + n * K];
				}

				p_re[m + n * M] = re;
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_prod_simple_gold)
	->Args({ 1024, 1024, 1024 })
	->Args({ 1, 9, 112 * 112 })
	->Args({ 64, 32, 112 * 112 })
	->Args({ 1024, 1024, 7 * 7})
	->UseRealTime();

void bm_prod_block_gold(benchmark::State &state) {
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	const int_t B = 32;

	while (state.KeepRunning()) {
		for (int_t n = 0; n < N; n += B) {
			for (int_t m = 0; m < M; m += B) {
				for (int_t k = 0; k < K; k += B) {

					for (int_t n_b = n; n_b < n + B; ++n_b) {
						for (int_t m_b = m; m_b < m + B; ++m_b) {
							auto re = 1.0f;
							for (int_t k_b = k; k_b < k + B; ++k_b) {
								 re += p_lhs[m_b * M + k_b] * p_rhs[n_b * N + k_b];
							}
							p_re[m_b + n_b * M] += re;
						}
					}


				}
			}
		}


#ifdef __linux__
		benchmark::ClobberMemory();
#endif
	}

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_prod_block_gold)->Args({ 1024, 1024, 1024 })->UseRealTime();

void bm_prod(benchmark::State &state){
	matrix<float, last_major_layout<2>> ts_lhs(pointi<2>{state.range(0), state.range(1)});
	matrix<float> ts_rhs(pointi<2>{state.range(1), state.range(2)});
	matrix<float> ts_output(pointi<2>{ts_lhs.shape()[0], ts_rhs.shape()[1]});
	fill(ts_lhs, 1.1f);
	fill(ts_rhs, 1.2f);

	while (state.KeepRunning()){
		auto lts_re = numeric::prod_general(ts_lhs, ts_rhs);
		copy(lts_re, ts_output);
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetItemsProcessed(state.iterations() * ts_lhs.shape()[0] * ts_lhs.shape()[1] * ts_rhs.shape()[1]);
}
BENCHMARK(bm_prod)
	->Args({ 1024, 1024, 1024 })
	->Args({ 1, 9, 112 * 112 })
	->Args({ 64, 32, 112 * 112 })
	->Args({ 1024, 1024, 7 * 7})
	->UseRealTime();

void bm_eigen_prod(benchmark::State &state){
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_lhs(state.range(0), state.range(1));
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_rhs(state.range(1), state.range(2));
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_output(state.range(0), state.range(2));
	mat_lhs.setRandom();
	mat_rhs.setRandom();

	while (state.KeepRunning()){
		mat_output = mat_lhs * mat_rhs;
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetItemsProcessed(state.iterations() * mat_lhs.rows() * mat_lhs.cols() * mat_rhs.cols());
}
BENCHMARK(bm_eigen_prod)
	->Args({ 1024, 1024, 1024 })
	->Args({ 1, 9, 112 * 112 })
	->Args({ 64, 32, 112 * 112 })
	->Args({ 1024, 1024, 7 * 7})
	->UseRealTime();


void bm_prod_block_static_gold(benchmark::State &state) {
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	constexpr int_t B = 32;

	while (state.KeepRunning()) {
		for (int_t n = 0; n < N; n += B) {
			for (int_t m = 0; m < M; m += B) {
				for (int_t k = 0; k < K; k += B) {

					float lhs[B * B];
					float rhs[B * B];
					float re[B * B];

					// _mm_prefetch(lhs, _MM_HINT_T0);
					// _mm_prefetch(rhs, _MM_HINT_T0);
					// auto lhs = (float *)aligned_alloc(B * B, 16);
					// auto rhs = (float *)aligned_alloc(B * B, 16);

					for (int_t n_b = 0; n_b < B; ++n_b) {
						for (int_t m_b = 0; m_b < B; ++m_b) {
							auto tmp = 1.0f;
							for (int_t k_b = 0; k_b < B; ++k_b) {
								tmp += lhs[m_b * B + k_b] * rhs[n_b * B + k_b];
							}
							re[n_b * B + m_b] += tmp;
						}
					}

					for (int_t n_b = 0; n_b < B; ++n_b) {
						for (int_t m_b = 0; m_b < B; ++m_b) {
							p_re[(n+n_b) * K + m + m_b] += re[m_b + n_b * B];
						}
					}



				}
			}
		}


#ifdef __linux__
		benchmark::ClobberMemory();
#endif
	}

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_prod_block_static_gold)->Args({ 1024, 1024, 1024 })->UseRealTime();

//void bm_prod_packed_sse_gold(benchmark::State &state) {
//	auto M = state.range(0);
//	auto K = state.range(1);
//	auto N = state.range(2);
//	auto p_lhs = new float[K * M];
//	auto p_rhs = new float[K * N];
//	auto p_re = new float[M * N];
//	//float *p_lhs = (float *)_aligned_malloc(K * M * 4, 16);
//	//float *p_rhs = (float *)_aligned_malloc(K * N * 4, 16);
//	//float *p_re = (float *)_aligned_malloc(M * N * 4, 16);
//
//	while (state.KeepRunning()) {
//		for (int_t n = 0; n < N; n += 4) {
//			for (int_t m = 0; m < M; m += 4) {
//				__m128 re = _mm_setzero_ps();
//				for (int_t k = 0; k < K; ++k) {
//					__m128 lhs_e = _mm_set_ps(p_lhs[k + m * K], p_lhs[k + 1 + m * K], p_lhs[k + 2 + m * K], p_lhs[k + 3 + m * K]);
//					__m128 rhs_e = _mm_set_ps(p_rhs[k + n * K], p_rhs[k + 1 + n * K], p_rhs[k + 2 + n * K], p_rhs[k + 3 + n * K]);
//					re = _mm_add_ps(re, _mm_mul_ps(lhs_e, rhs_e));
//					//re += p_lhs[k + m * K] * p_rhs[k + n * K];
//				}
//
//				p_re[m + 0 + (n + 0) * M] = re.m128_f32[0];
//				p_re[m + 1 + (n + 1) * M] = re.m128_f32[0];
//				p_re[m + 2 + (n + 2) * M] = re.m128_f32[0];
//				p_re[m + 3 + (n + 3) * M] = re.m128_f32[0];
//			}
//		}
//#ifdef __linux__
//		benchmark::ClobberMemory();
//#endif
//	}
//
//	state.SetItemsProcessed(state.iterations() * M * K * N);
//}
//BENCHMARK(bm_prod_packed_sse_gold)->Args({ 1024, 1024, 1024 })->UseRealTime();

//void bm_prod_simple_sse_gold(benchmark::State &state){
//	auto M = state.range(0);
//	auto K = state.range(1);
//	auto N = state.range(2);
//	 auto p_lhs = new float[K * M];
//	 auto p_rhs = new float[K * N];
//	 auto p_re = new float[M * N];
//	//float *p_lhs = (float *)aligned_alloc(16, K * M * 4);
//	//float *p_rhs = (float *)aligned_alloc(16, K * N * 4);
//	//float *p_re = (float *)aligned_alloc(16, M * N * 4);
//
//	while (state.KeepRunning()){
//		for (int_t n = 0; n < N; ++n){
//			for (int_t m = 0; m < M; ++m){
//				auto re = _mm_setzero_ps();
//				for (int_t k = 0; k < K; k += 4){
//					re = _mm_add_ps(re, _mm_mul_ps(*((__m128 *)(p_lhs + k + m * K)), * ((__m128 *)(p_rhs + k + n * K))));
//				}
//				p_re[m + n * M] = re[0] + re[1] + re[2] + re[3];
//			}
//		}
//	#ifdef __linux__
//		benchmark::ClobberMemory();
//	#endif
//	}
//
//	state.SetItemsProcessed(state.iterations() * M * K * N);
//}
//BENCHMARK(bm_prod_simple_sse_gold)->Args({2048, 32, 2048})->UseRealTime();

//void bm_prod_simple_sse_block(benchmark::State &state){
//	auto M = state.range(0);
//	auto K = state.range(1);
//	auto N = state.range(2);
//	// auto p_lhs = new float[K * M];
//	// auto p_rhs = new float[K * N];
//	// auto p_re = new float[M * N];
//	float *p_lhs = (float *)aligned_alloc(16, K * M * 4);
//	float *p_rhs = (float *)aligned_alloc(16, K * N * 4);
//	float *p_re = (float *)aligned_alloc(16, M * N * 4);
//
//	while (state.KeepRunning()){
//		for (int_t n = 0; n < N; ++n){
//			for (int_t m = 0; m < M; ++m){
//				auto re = _mm_setzero_ps();
//				for (int_t k = 0; k < K; k += 4){
//					re = _mm_add_ps(re, _mm_mul_ps(*((__m128 *)(p_lhs + k + m * K)), * ((__m128 *)(p_rhs + k + n * K))));
//				}
//				p_re[m + n * M] = re[0] + re[1] + re[2] + re[3];
//			}
//		}
//	#ifdef __linux__
//		benchmark::ClobberMemory();
//	#endif
//	}
//
//	state.SetItemsProcessed(state.iterations() * M * K * N);
//}
//BENCHMARK(bm_prod_simple_sse_gold)->Args({2048, 32, 2048})->UseRealTime();

static void sgemm(int m, int n, int d, float *A, float *C)
{
	const int STRIDE = 40;
	const int blocksize = 120;       /*40, 40*/




	{
		register __m128 cmat0;
		register __m128 cmat1;
		register __m128 cmat2;
		register __m128 cmat3;
		register __m128 cmat4;

		register __m128 cmat5;
		register __m128 cmat6;
		register __m128 cmat7;
		register __m128 cmat8;
		register __m128 cmat9;
		__m128 amat0, amat1;
		float* sum1;
		float* sum2;

		/*for(int j1 = 0; j1 < n; j1 += blocksize) {*/
		for (int j = 0; j < n; j++) {
			for (int i1 = 0; i1 < n; i1 += blocksize) {
				for (int k1 = 0; k1 < m; k1 += blocksize) {


					for (int i = i1; i < i1 + blocksize && i < n / STRIDE*STRIDE; i += STRIDE) {
						sum1 = C + i + j*n;
						cmat0 = _mm_loadu_ps(sum1);
						cmat1 = _mm_loadu_ps(sum1 + 4);
						cmat2 = _mm_loadu_ps(sum1 + 8);
						cmat3 = _mm_loadu_ps(sum1 + 12);
						cmat4 = _mm_loadu_ps(sum1 + 16);
						cmat5 = _mm_loadu_ps(sum1 + 20);
						cmat6 = _mm_loadu_ps(sum1 + 24);
						cmat7 = _mm_loadu_ps(sum1 + 28);
						cmat8 = _mm_loadu_ps(sum1 + 32);
						cmat9 = _mm_loadu_ps(sum1 + 36);
						for (int k = k1; k < k1 + blocksize && k < m; k++) {

							amat1 = _mm_load1_ps(A + j*(n + 1) + k*n);

							sum2 = A + i + k*n;

							amat0 = _mm_loadu_ps(sum2);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat0 = _mm_add_ps(cmat0, amat0);

							amat0 = _mm_loadu_ps(sum2 + 4);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat1 = _mm_add_ps(cmat1, amat0);

							amat0 = _mm_loadu_ps(sum2 + 8);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat2 = _mm_add_ps(cmat2, amat0);

							amat0 = _mm_loadu_ps(sum2 + 12);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat3 = _mm_add_ps(cmat3, amat0);

							amat0 = _mm_loadu_ps(sum2 + 16);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat4 = _mm_add_ps(cmat4, amat0);

							amat0 = _mm_loadu_ps(sum2 + 20);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat5 = _mm_add_ps(cmat5, amat0);

							amat0 = _mm_loadu_ps(sum2 + 24);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat6 = _mm_add_ps(cmat6, amat0);

							amat0 = _mm_loadu_ps(sum2 + 28);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat7 = _mm_add_ps(cmat7, amat0);

							amat0 = _mm_loadu_ps(sum2 + 32);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat8 = _mm_add_ps(cmat8, amat0);

							amat0 = _mm_loadu_ps(sum2 + 36);
							amat0 = _mm_mul_ps(amat0, amat1);
							cmat9 = _mm_add_ps(cmat9, amat0);


						}
						_mm_storeu_ps(sum1, cmat0);
						_mm_storeu_ps(sum1 + 4, cmat1);
						_mm_storeu_ps(sum1 + 8, cmat2);
						_mm_storeu_ps(sum1 + 12, cmat3);
						_mm_storeu_ps(sum1 + 16, cmat4);

						_mm_storeu_ps(sum1 + 20, cmat5);
						_mm_storeu_ps(sum1 + 24, cmat6);
						_mm_storeu_ps(sum1 + 28, cmat7);
						_mm_storeu_ps(sum1 + 32, cmat8);
						_mm_storeu_ps(sum1 + 36, cmat9);

					}




				}
			}






			for (int k1 = 0; k1 < m; k1 += blocksize) {

				for (int i = n / STRIDE*STRIDE; i < n / 4 * 4; i += 4) {
					cmat0 = _mm_loadu_ps(C + i + j*n);
					for (int k = k1; k < k1 + blocksize && k < m; k++) {
						amat1 = _mm_load1_ps(A + j*(n + 1) + k*n);

						amat0 = _mm_loadu_ps(A + i + k*n);
						amat0 = _mm_mul_ps(amat0, amat1);
						cmat0 = _mm_add_ps(cmat0, amat0);
					}
					_mm_storeu_ps(C + i + j*n, cmat0);
				}
				for (int i = n / 4 * 4; i < n; i++) {
					for (int k = k1; k < k1 + blocksize && k < m; k++) {
						C[i + j*n] += A[i + k*n] * A[j*(n + 1) + k*n];
					}
				}

			}


		}
	}
}

void bm_prod_block_sgemm(benchmark::State &state) {
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M + K * N];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	const int_t B = 32;

	while (state.KeepRunning()) {
		sgemm(M, N, K, p_lhs, p_re);

#ifdef __linux__
		benchmark::ClobberMemory();
#endif
	}

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_prod_block_sgemm)->Args({ 1024, 1024, 1024 })->UseRealTime();
