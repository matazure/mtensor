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
	->Args({ 3, 9, 112 * 112})
	->Args({ 1, 9, 112 * 112 })
	->Args({ 64, 32, 112 * 112 })
	->Args({ 1024, 1024, 7 * 7})
	->Args({ 1000, 1024, 1})
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
	->Args({ 1000, 1024, 1})
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
	->Args({ 1000, 1024, 1})
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
