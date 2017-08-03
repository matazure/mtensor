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

void bm_tn_nmk_prod_simple_gold(benchmark::State &state){
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
					re += p_lhs[m * K + k] * p_rhs[k + n * K];
				}

				p_re[m + n * M] = re;
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	delete[] p_lhs;
	delete[] p_rhs;
	delete[] p_re;

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_tn_nmk_prod_simple_gold)
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	///mobilenet_0.5
	//general conv
	->Args({ 112 * 112, 9, 3})
	//depthwise conv
	->Args({ 112 * 112, 9 ,1})
	->Args({ 56 * 56, 9, 1})
	->Args({ 28 * 28, 9, 1})
	->Args({ 14 * 14, 9, 1})
	->Args({ 7 * 7, 9, 1})
	//pointwise conv
	->Args({ 112 * 112, 16, 32})
	->Args({ 56 * 56, 32, 64})
	->Args({ 56 * 56, 64, 64})
	->Args({ 28 * 28, 64, 128})
	->Args({ 28 * 28, 128, 128})
	->Args({ 14 * 14, 128, 128})
	->Args({ 14 * 14, 128, 256})
	->Args({ 14 * 14, 256, 256})
	->Args({ 7 * 7, 256, 512})
	->Args({ 7 * 7, 512, 512})
	->Args({ 7 * 7, 512, 30})
	->UseRealTime();

void bm_nn_nmk_prod_simple(benchmark::State &state){
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
					re += p_lhs[m + M * k] * p_rhs[k + K * n];
				}

				p_re[m + n * M] = re;
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	delete[] p_lhs;
	delete[] p_rhs;
	delete[] p_re;

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_nn_nmk_prod_simple)
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	///mobilenet_0.5
	//general conv
	->Args({ 112 * 112, 9, 3})
	//depthwise conv
	->Args({ 112 * 112, 9 ,1})
	->Args({ 56 * 56, 9, 1})
	->Args({ 28 * 28, 9, 1})
	->Args({ 14 * 14, 9, 1})
	->Args({ 7 * 7, 9, 1})
	//pointwise conv
	->Args({ 112 * 112, 16, 32})
	->Args({ 56 * 56, 32, 64})
	->Args({ 56 * 56, 64, 64})
	->Args({ 28 * 28, 64, 128})
	->Args({ 28 * 28, 128, 128})
	->Args({ 14 * 14, 128, 128})
	->Args({ 14 * 14, 128, 256})
	->Args({ 14 * 14, 256, 256})
	->Args({ 7 * 7, 256, 512})
	->Args({ 7 * 7, 512, 512})
	->Args({ 7 * 7, 512, 30})
	->UseRealTime();

void bm_nn_nkm_prod_simple(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	while (state.KeepRunning()){
		for (int_t n = 0; n < N; ++n){
			for (int_t k = 0; k < K; ++k){
				for (int_t m = 0; m < M; ++m){
					p_re[m + n * M] += p_lhs[m + M * k] * p_rhs[k + K * n];
				}
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	delete[] p_lhs;
	delete[] p_rhs;
	delete[] p_re;

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_nn_nkm_prod_simple)
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	///mobilenet_0.5
	//general conv
	->Args({ 112 * 112, 9, 3})
	//depthwise conv
	->Args({ 112 * 112, 9 ,1})
	->Args({ 56 * 56, 9, 1})
	->Args({ 28 * 28, 9, 1})
	->Args({ 14 * 14, 9, 1})
	->Args({ 7 * 7, 9, 1})
	//pointwise conv
	->Args({ 112 * 112, 16, 32})
	->Args({ 56 * 56, 32, 64})
	->Args({ 56 * 56, 64, 64})
	->Args({ 28 * 28, 64, 128})
	->Args({ 28 * 28, 128, 128})
	->Args({ 14 * 14, 128, 128})
	->Args({ 14 * 14, 128, 256})
	->Args({ 14 * 14, 256, 256})
	->Args({ 7 * 7, 256, 512})
	->Args({ 7 * 7, 512, 512})
	->Args({ 7 * 7, 512, 30})
	->UseRealTime();

void bm_nn_nkm_prod_simple_sse(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	while (state.KeepRunning()){
		for (int_t n = 0; n < N; n += 4){
			for (int_t k = 0; k < K; k += 4){
				for (int_t m = 0; m < M;  m += 4){
					register auto re_v000 = _mm_setzero_ps();
					register auto re_v010 = _mm_setzero_ps();
					register auto re_v020 = _mm_setzero_ps();
					register auto re_v030 = _mm_setzero_ps();


				#define MULTIPLY_4x1(_m, _n, _k) \
					register auto rhs_v##_m##_n##_k = _mm_set_ps(p_rhs[k + _k + K * (n + _n)], p_rhs[k + _k + K * (n + _n)], p_rhs[k + _k+ K * (n + _n)], p_rhs[k + _k + K * (n + _n)]); \
					register auto lhs_v_p##_m##_n##_k =  _mm_load_ps(p_lhs + m + M * (k + _k) + _m * 4); \
					re_v##_m##_n##0 = _mm_add_ps(re_v##_m##_n##0, _mm_mul_ps(lhs_v_p##_m##_n##_k, rhs_v##_m##_n##_k));

				#define STORE_RE(_m, _n, _k) \
					_mm_store_ps(p_re + m + 4 * _m + (n + _n) * M, _mm_add_ps(_mm_load_ps(p_re + m + 4 * _m + (n + _n) * M), re_v##_m##_n##0))

					MULTIPLY_4x1(0, 0, 0);
					MULTIPLY_4x1(0, 0, 1);
					MULTIPLY_4x1(0, 0, 2);
					MULTIPLY_4x1(0, 0, 3);

					STORE_RE(0, 0, NONE);


					MULTIPLY_4x1(0, 1, 0);
					MULTIPLY_4x1(0, 1, 1);
					MULTIPLY_4x1(0, 1, 2);
					MULTIPLY_4x1(0, 1, 3);

					STORE_RE(0, 1, NONE);


					MULTIPLY_4x1(0, 2, 0);
					MULTIPLY_4x1(0, 2, 1);
					MULTIPLY_4x1(0, 2, 2);
					MULTIPLY_4x1(0, 2, 3);

					STORE_RE(0, 2, NONE);


					MULTIPLY_4x1(0, 3, 0);
					MULTIPLY_4x1(0, 3, 1);
					MULTIPLY_4x1(0, 3, 2);
					MULTIPLY_4x1(0, 3, 3);

					STORE_RE(0, 3, NONE);



					//
					// MULTIPLY_4x1(1, 0, 0);
					// MULTIPLY_4x1(1, 0, 1);
					// MULTIPLY_4x1(1, 0, 2);
					// MULTIPLY_4x1(1, 0, 3);
					//
					// MULTIPLY_4x1(1, 1, 0);
					// MULTIPLY_4x1(1, 1, 1);
					// MULTIPLY_4x1(1, 1, 2);
					// MULTIPLY_4x1(1, 1, 3);
					//
					// MULTIPLY_4x1(1, 2, 0);
					// MULTIPLY_4x1(1, 2, 1);
					// MULTIPLY_4x1(1, 2, 2);
					// MULTIPLY_4x1(1, 2, 3);
					//
					// MULTIPLY_4x1(1, 3, 0);
					// MULTIPLY_4x1(1, 3, 1);
					// MULTIPLY_4x1(1, 3, 2);
					// MULTIPLY_4x1(1, 3, 3);
				}
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	delete[] p_lhs;
	delete[] p_rhs;
	delete[] p_re;

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_nn_nkm_prod_simple_sse)
	->Args({ 16, 16, 16 })
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	///mobilenet_0.5
	//general conv
	// ->Args({ 112 * 112, 9, 3})
	// //depthwise conv
	// ->Args({ 112 * 112, 9 ,1})
	// ->Args({ 56 * 56, 9, 1})
	// ->Args({ 28 * 28, 9, 1})
	// ->Args({ 14 * 14, 9, 1})
	// ->Args({ 7 * 7, 9, 1})
	//pointwise conv
	->Args({ 112 * 112, 16, 32})
	->Args({ 56 * 56, 32, 64})
	->Args({ 56 * 56, 64, 64})
	->Args({ 28 * 28, 64, 128})
	->Args({ 28 * 28, 128, 128})
	// ->Args({ 14 * 14, 128, 128})
	// ->Args({ 14 * 14, 128, 256})
	// ->Args({ 14 * 14, 256, 256})
	// ->Args({ 7 * 7, 256, 512})
	// ->Args({ 7 * 7, 512, 512})
	// ->Args({ 7 * 7, 512, 30})
	->UseRealTime();

void bm_tn_nmk_block_nmk_prod_gold(benchmark::State &state) {
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	const int_t B = 16;

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

	delete[] p_lhs;
	delete[] p_rhs;
	delete[] p_re;

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_tn_nmk_block_nmk_prod_gold)
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	->UseRealTime();

void bm_tn_prod(benchmark::State &state){
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
//!bm_args
BENCHMARK(bm_tn_prod)
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	///mobilenet_0.5
	//general conv
	->Args({ 112 * 112, 9, 3})
	//depthwise conv
	->Args({ 112 * 112, 9 ,1})
	->Args({ 56 * 56, 9, 1})
	->Args({ 28 * 28, 9, 1})
	->Args({ 14 * 14, 9, 1})
	->Args({ 7 * 7, 9, 1})
	//pointwise conv
	->Args({ 112 * 112, 16, 32})
	->Args({ 56 * 56, 32, 64})
	->Args({ 56 * 56, 64, 64})
	->Args({ 28 * 28, 64, 128})
	->Args({ 28 * 28, 128, 128})
	->Args({ 14 * 14, 128, 128})
	->Args({ 14 * 14, 128, 256})
	->Args({ 14 * 14, 256, 256})
	->Args({ 7 * 7, 256, 512})
	->Args({ 7 * 7, 512, 512})
	->Args({ 7 * 7, 512, 30})
	->UseRealTime();

void bm_nn_eigen_prod(benchmark::State &state){
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
BENCHMARK(bm_nn_eigen_prod)
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	///mobilenet_0.5
	//general conv
	->Args({ 112 * 112, 9, 3})
	//depthwise conv
	->Args({ 112 * 112, 9 ,1})
	->Args({ 56 * 56, 9, 1})
	->Args({ 28 * 28, 9, 1})
	->Args({ 14 * 14, 9, 1})
	->Args({ 7 * 7, 9, 1})
	//pointwise conv
	->Args({ 112 * 112, 16, 32})
	->Args({ 56 * 56, 32, 64})
	->Args({ 56 * 56, 64, 64})
	->Args({ 28 * 28, 64, 128})
	->Args({ 28 * 28, 128, 128})
	->Args({ 14 * 14, 128, 128})
	->Args({ 14 * 14, 128, 256})
	->Args({ 14 * 14, 256, 256})
	->Args({ 7 * 7, 256, 512})
	->Args({ 7 * 7, 512, 512})
	->Args({ 7 * 7, 512, 30})
	->UseRealTime();

void bm_tn_eigen_prod(benchmark::State &state){
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_lhs(state.range(0), state.range(1));
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
BENCHMARK(bm_tn_eigen_prod)
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	///mobilenet_0.5
	//general conv
	->Args({ 112 * 112, 9, 3})
	//depthwise conv
	->Args({ 112 * 112, 9 ,1})
	->Args({ 56 * 56, 9, 1})
	->Args({ 28 * 28, 9, 1})
	->Args({ 14 * 14, 9, 1})
	->Args({ 7 * 7, 9, 1})
	//pointwise conv
	->Args({ 112 * 112, 16, 32})
	->Args({ 56 * 56, 32, 64})
	->Args({ 56 * 56, 64, 64})
	->Args({ 28 * 28, 64, 128})
	->Args({ 28 * 28, 128, 128})
	->Args({ 14 * 14, 128, 128})
	->Args({ 14 * 14, 128, 256})
	->Args({ 14 * 14, 256, 256})
	->Args({ 7 * 7, 256, 512})
	->Args({ 7 * 7, 512, 512})
	->Args({ 7 * 7, 512, 30})
	->UseRealTime();

// void bm_prod_block_static_gold(benchmark::State &state) {
// 	auto M = state.range(0);
// 	auto K = state.range(1);
// 	auto N = state.range(2);
// 	auto p_lhs = new float[K * M];
// 	auto p_rhs = new float[K * N];
// 	auto p_re = new float[M * N];
//
// 	constexpr int_t B = 32;
//
// 	while (state.KeepRunning()) {
// 		for (int_t n = 0; n < N; n += B) {
// 			for (int_t m = 0; m < M; m += B) {
// 				for (int_t k = 0; k < K; k += B) {
//
// 					float lhs[B * B];
// 					float rhs[B * B];
// 					float re[B * B];
//
// 					// _mm_prefetch(lhs, _MM_HINT_T0);
// 					// _mm_prefetch(rhs, _MM_HINT_T0);
// 					// auto lhs = (float *)aligned_alloc(B * B, 16);
// 					// auto rhs = (float *)aligned_alloc(B * B, 16);
//
// 					for (int_t n_b = 0; n_b < B; ++n_b) {
// 						for (int_t m_b = 0; m_b < B; ++m_b) {
// 							auto tmp = 1.0f;
// 							for (int_t k_b = 0; k_b < B; ++k_b) {
// 								tmp += lhs[m_b * B + k_b] * rhs[n_b * B + k_b];
// 							}
// 							re[n_b * B + m_b] += tmp;
// 						}
// 					}
//
// 					for (int_t n_b = 0; n_b < B; ++n_b) {
// 						for (int_t m_b = 0; m_b < B; ++m_b) {
// 							p_re[(n+n_b) * K + m + m_b] += re[m_b + n_b * B];
// 						}
// 					}
//
//
//
// 				}
// 			}
// 		}
//
//
// #ifdef __linux__
// 		benchmark::ClobberMemory();
// #endif
// 	}
//
// 	state.SetItemsProcessed(state.iterations() * M * K * N);
// }
// BENCHMARK(bm_prod_block_static_gold)->Args({ 1024, 1024, 1024 })->UseRealTime();

// void bm_prod_4x4_sse(benchmark::State &state) {
// 	__m128 rhs[4];
// 	__m128 lhs[4];
// 	__m128 re[4];
//
// 	while (state.KeepRunning()) {
// 		re = _mm_mul_ps(rhs)
// #ifdef __linux__
// 		benchmark::ClobberMemory();
// #endif
// 	}
//
// 	state.SetItemsProcessed(state.iterations() * M * K * N);
// }
// BENCHMARK(bm_prod_4x4_sse)->UseRealTime();

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
    //
    // auto p_X = X.template data<float>();
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
