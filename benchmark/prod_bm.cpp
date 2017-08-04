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

void bm_tn_nmk_prod(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	// auto p_lhs = new float[K * M];
	// auto p_rhs = new float[K * N];
	// auto p_re = new float[M * N];
	matrix<float, row_major_layout> mat_lhs(M, K);
	matrix<float> mat_rhs(K, N);
	matrix<float> mat_re(M, N);

	while (state.KeepRunning()){
		for (int_t n = 0; n < N; ++n){
			for (int_t m = 0; m < M; ++m){
				float re = 0.0f;
				for (int_t k = 0; k < K; ++k){
					re += mat_lhs(m, k) * mat_rhs(k, n);
				}

				mat_re(m, n) = re;
			}
		}
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_tn_nmk_prod)
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

void bm_tn_nmk_prod_k4_sse(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	while (state.KeepRunning()){
		for (int_t n = 0; n < N; ++n){
			for (int_t m = 0; m < M; ++m){
				__m128 re = _mm_setzero_ps();
				for (int_t k = 0; k < K; k += 4){
					re = _mm_add_ps(re, _mm_mul_ps(_mm_load_ps(p_lhs + m*K + k), _mm_load_ps(p_rhs + n * K + k)));
				}

				auto tmp0 = _mm_hadd_ps(re, re);
				auto tmp1 = _mm_hadd_ps(tmp0, tmp0);
				p_re[m + n * M] += tmp1[0];
				// p_re[m + n * M] += re[0];
				// p_re[m + n * M] += re[1];
				// p_re[m + n * M] += re[3];
				// p_re[m + n * M] += re[2];
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
BENCHMARK(bm_tn_nmk_prod_k4_sse)
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
	->Args({ 14 * 14, 128, 128})
	->Args({ 14 * 14, 128, 256})
	->Args({ 14 * 14, 256, 256})
	->Args({ 7 * 7, 256, 512})
	->Args({ 7 * 7, 512, 512})
	->Args({ 7 * 7, 512, 30})
	->UseRealTime();

void bm_nn_nkm_prod_n4k4m4_sse(benchmark::State &state){
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
BENCHMARK(bm_nn_nkm_prod_n4k4m4_sse)
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

void bm_t4n_mnk_prod_m4n4k4_sse(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);
	auto p_lhs = new float[K * M];
	auto p_rhs = new float[K * N];
	auto p_re = new float[M * N];

	while (state.KeepRunning()){
		for (int_t m = 0; m < M;  m += 4){
			for (int_t n = 0; n < N; n += 4){
				for (int_t k = 0; k < K; k += 4){
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
BENCHMARK(bm_t4n_mnk_prod_m4n4k4_sse)
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
