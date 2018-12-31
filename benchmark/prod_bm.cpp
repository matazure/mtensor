#define EIGEN_DONT_PARALLELIZE

#include <bm_config.hpp>
#include <matazure/tensor>
 #include <Eigen/Core>
 #include <Eigen/Dense>


using namespace matazure;

void bm_gold_tn_nmk_prod_simple(benchmark::State &state){
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
BENCHMARK(bm_gold_tn_nmk_prod_simple)
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

void bm_gold_tn_nmk_block_nmk_prod(benchmark::State &state) {
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
BENCHMARK(bm_gold_tn_nmk_block_nmk_prod)
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
		auto lts_re = puzzle::prod_general(ts_lhs, ts_rhs);
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
 		mat_output.noalias() = mat_lhs * mat_rhs;
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
 	//mobilenet_0.5
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
	 ->Args({ 1024, 8, 1 })
	 ->Args({ 1024, 4, 1 })
 	->UseRealTime();

 void bm_tn_eigen_prod(benchmark::State &state){
 	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_lhs(state.range(0), state.range(1));
 	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_rhs(state.range(1), state.range(2));
 	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_output(state.range(0), state.range(2));
 	mat_lhs.setRandom();
 	mat_rhs.setRandom();

 	while (state.KeepRunning()){
 		mat_output.noalias() = mat_lhs * mat_rhs;
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

#ifdef MATAZURE_SSE

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
				__m128 re0 = _mm_setzero_ps();
				__m128 re1 = _mm_setzero_ps();

				for (int_t k = 0; k < K; k += 8){
					re0 = _mm_add_ps(re0, _mm_mul_ps(_mm_load_ps(p_lhs + m*K + k + 0), _mm_load_ps(p_rhs + n * K + k + 0)));
					re1 = _mm_add_ps(re1, _mm_mul_ps(_mm_load_ps(p_lhs + m*K + k + 4), _mm_load_ps(p_rhs + n * K + k + 4)));
					// re2 = _mm_add_ps(re2, _mm_mul_ps(_mm_load_ps(p_lhs + m*K + k + 8), _mm_load_ps(p_rhs + n * K + k + 8)));
					// re3 = _mm_add_ps(re3, _mm_mul_ps(_mm_load_ps(p_lhs + m*K + k + 12), _mm_load_ps(p_rhs + n * K + k + 12)));
				}

				auto tmp = _mm_add_ps(re0, re1);
				//tmp = _mm_hadd_ps(tmp, tmp);
				//tmp = _mm_hadd_ps(tmp, tmp);
				// auto tmp0 = _mm_hadd_ps(re0, re1);
				//
				// auto tmp1 = _mm_hadd_ps(tmp0, tmp0);
			#if defined(__clang__) || defined(__GNUC__)
				p_re[m + n * M] += tmp[0];
			#else
				p_re[m + n * M] += tmp.m128_f32[0];
			#endif
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
	// ->Args({1, 1024 * 1024, 1})
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
					register auto re_v00None = _mm_setzero_ps();
					register auto re_v01None = _mm_setzero_ps();
					register auto re_v02None = _mm_setzero_ps();
					register auto re_v03None = _mm_setzero_ps();

				#define MULTIPLY_4x1(_m, _n, _k) \
					register auto rhs_v##_m##_n##_k = _mm_set_ps(p_rhs[k + _k + K * (n + _n)], p_rhs[k + _k + K * (n + _n)], p_rhs[k + _k+ K * (n + _n)], p_rhs[k + _k + K * (n + _n)]); \
					register auto lhs_v_p##_m##_n##_k =  _mm_load_ps(p_lhs + m + M * (k + _k) + _m * 4); \
					re_v##_m##_n##None = _mm_add_ps(re_v##_m##_n##None, _mm_mul_ps(lhs_v_p##_m##_n##_k, rhs_v##_m##_n##_k));

				#define STORE_RE(_m, _n, _k) \
					_mm_store_ps(p_re + m + 4 * _m + (n + _n) * M, _mm_add_ps(_mm_load_ps(p_re + m + 4 * _m + (n + _n) * M), re_v##_m##_n##None))

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

				#undef MULTIPLY_4x1
				#undef STORE_RE
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
	->Args({ 8 * 8, 512, 32})
	->UseRealTime();

void bm_t4n_mnk_prod_m4n4k4_sse(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);


	while (state.KeepRunning()){
		auto p_lhs = new float[K * M];
		auto p_rhs = new float[K * N];
		auto p_re = new float[M * N];

		for (auto  p_rhs_prefetch = p_rhs; p_rhs_prefetch < (p_rhs + N * K); p_rhs_prefetch += 16){
			_mm_prefetch((char *)p_rhs, _MM_HINT_NTA);
		}

		for (int_t m = 0; m < M;  m += 4){
			for (int_t n = 0; n < N; n += 4){
				 auto re_v00None = _mm_setzero_ps();
				 auto re_v01None = _mm_setzero_ps();
				 auto re_v02None = _mm_setzero_ps();
				 auto re_v03None = _mm_setzero_ps();

				//  auto p_lhs_tmp = p_lhs + 0 * 4 + m * K;
				//  auto p_rhs_tmp = p_rhs + 0 * 4 + K * n;
				//  _mm_prefetch(p_lhs_tmp + 16, _MM_HINT_T1);
				//  _mm_prefetch(p_rhs_tmp + K, _MM_HINT_T1);


				for (int_t k = 0; k < K; k += 4){
					// auto p_lhs_tmp = p_lhs + k * 4 + m * K;
					// auto p_rhs_tmp = p_rhs + k * 4 + K * n;
					//  _mm_prefetch(p_lhs_tmp + 16, _MM_HINT_T0);
					//  _mm_prefetch(p_lhs_tmp + 16, _MM_HINT_T0);

				#define MULTIPLY_4x1(_m, _n, _k) \
					auto lhs_v_p##_m##_n##_k =  _mm_load_ps(p_lhs + (m + 4 * _m) * K + (k + _k) * 4); \
					auto rhs_v##_m##_n##_k = _mm_set_ps(p_rhs[k + _k + K * (n + _n)], p_rhs[k + _k + K * (n + _n)], p_rhs[k + _k+ K * (n + _n)], p_rhs[k + _k + K * (n + _n)]); \
					re_v##_m##_n##None = _mm_add_ps(re_v##_m##_n##None, _mm_mul_ps(lhs_v_p##_m##_n##_k, rhs_v##_m##_n##_k));

					MULTIPLY_4x1(0, 0, 0);
					MULTIPLY_4x1(0, 0, 1);
					MULTIPLY_4x1(0, 0, 2);
					MULTIPLY_4x1(0, 0, 3);

					MULTIPLY_4x1(0, 1, 0);
					MULTIPLY_4x1(0, 1, 1);
					MULTIPLY_4x1(0, 1, 2);
					MULTIPLY_4x1(0, 1, 3);

					MULTIPLY_4x1(0, 2, 0);
					MULTIPLY_4x1(0, 2, 1);
					MULTIPLY_4x1(0, 2, 2);
					MULTIPLY_4x1(0, 2, 3);

					MULTIPLY_4x1(0, 3, 0);
					MULTIPLY_4x1(0, 3, 1);
					MULTIPLY_4x1(0, 3, 2);
					MULTIPLY_4x1(0, 3, 3);
				}

				#define STORE_RE(_m, _n, _k) \
					_mm_store_ps(p_re + m + 4 * _m + (n + _n) * M, re_v##_m##_n##None)

				STORE_RE(0, 0, None);
				STORE_RE(0, 1, None);
				STORE_RE(0, 2, None);
				STORE_RE(0, 3, None);
			}
		}

		delete[] p_lhs;
		delete[] p_rhs;
		delete[] p_re;

	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}



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
	->Args({ 8 * 8, 512, 32})
	->UseRealTime();

void bm_t4n_mnk_prod_m4n4k4_sse_seq(benchmark::State &state){
	auto M = state.range(0);
	auto K = state.range(1);
	auto N = state.range(2);


	while (state.KeepRunning()){
		auto p_lhs = new float[K * M];
		auto p_rhs = new float[K * N];
		auto p_re = new float[M * N];

		for (auto  p_rhs_prefetch = p_rhs; p_rhs_prefetch < (p_rhs + N * K); p_rhs_prefetch += 16){
			_mm_prefetch((char *)p_rhs, _MM_HINT_NTA);
		}

		for (int_t m = 0; m < M;  m += 4){
			for (int_t n = 0; n < N; n += 4){
				auto re_0 = _mm_setzero_ps();
				auto re_1 = _mm_setzero_ps();
				auto re_2 = _mm_setzero_ps();
				auto re_3 = _mm_setzero_ps();

				auto p_lhs_tmp = p_lhs + m * K;
				auto p_rhs_tmp = p_rhs + n * K;
				auto p_re_tmp = p_re + n * 4 + m * N;

				for (int_t k = 0; k < K; k += 4){
					auto lhs0 = _mm_load_ps(p_lhs_tmp);
					p_lhs_tmp += 4;
					auto lhs1 = _mm_load_ps(p_lhs_tmp);
					p_lhs_tmp += 4;
					auto lhs2 = _mm_load_ps(p_lhs_tmp);
					p_lhs_tmp += 4;
					auto lhs3 = _mm_load_ps(p_lhs_tmp);
					p_lhs_tmp += 4;

					auto rhs00 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_0 += lhs0 * rhs00;
					auto rhs10 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_0 += lhs1 * rhs10;
					auto rhs20 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_0 += lhs2 * rhs20;
					auto rhs30 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_0 += lhs3 * rhs30;

					auto rhs01 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_1 += lhs0 * rhs01;
					auto rhs11 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_1 += lhs1 * rhs11;
					auto rhs21 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_1 += lhs2 * rhs21;
					auto rhs31 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_1 += lhs3 * rhs31;

					auto rhs02 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_2 += lhs0 * rhs02;
					auto rhs12 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_2 += lhs1 * rhs12;
					auto rhs22 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_2 += lhs2 * rhs22;
					auto rhs32 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_2 += lhs3 * rhs32;

					auto rhs03 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_3 += lhs0 * rhs03;
					auto rhs13 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_3 += lhs1 * rhs13;
					auto rhs23 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_3 += lhs2 * rhs23;
					auto rhs33 = _mm_load_ps1(p_rhs_tmp);
					p_rhs_tmp += 1;
					re_3 += lhs3 * rhs33;
				}

				_mm_store_ps(p_re_tmp, re_0);
				p_re_tmp += 4;
				_mm_store_ps(p_re_tmp, re_1);
				p_re_tmp += 4;
				_mm_store_ps(p_re_tmp, re_2);
				p_re_tmp += 4;
				_mm_store_ps(p_re_tmp, re_3);
				p_re_tmp += 4;
			}
		}

		delete[] p_lhs;
		delete[] p_rhs;
		delete[] p_re;

	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}



	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_t4n_mnk_prod_m4n4k4_sse_seq)
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
	->Args({ 8 * 8, 512, 32})
	->UseRealTime();

#endif

#if defined(__GNUC__) &&  defined(MATAZURE_SSE)

void bm_t4n_mnk_prod_asm(benchmark::State &state){
	size_t M = state.range(0);
	size_t K = state.range(1);
	size_t N = state.range(2);
	 auto p_lhs = new float[K * M];
	 auto p_rhs = new float[K * N];
	 auto p_re = new float[M * N];

	p_re[0] =0;

	auto tmp = new int[3];
	tmp[0] = 0;

	while (state.KeepRunning()){
		for (int_t m = 0; m < M;  m += 4){
			for (int_t n = 0; n < N; n += 4){
				auto p_re_tmp = p_re + n * 4 + m * N;
				auto p_lhs_tmp = p_lhs + 0 * 4 + m * K;
				auto p_rhs_tmp = p_rhs + 0 * 4 + K * n;

				for (int_t k = 0; k < K; k += 4){
					asm volatile (
						"movaps (%[p_lhs_tmp]), %%xmm0\n\t"
						"movaps 16(%[p_lhs_tmp]), %%xmm1\n\t"
						"movaps 32(%[p_lhs_tmp]), %%xmm2\n\t"
						"movaps 48(%[p_lhs_tmp]), %%xmm3\n\t"
						"movq $48, %%rcx\n\t"
						"1:\n\t"
						"movss (%[p_rhs_tmp], %%rcx), %%xmm4\n\t"
						"shufps $0, %%xmm4, %%xmm4\n\t"
						"mulps %%xmm0, %%xmm4\n\t"
						"movaps %%xmm4, %%xmm5\n\t"
						"movss 4(%[p_rhs_tmp], %%rcx), %%xmm4\n\t"
						"shufps $0, %%xmm4, %%xmm4\n\t"
						"mulps %%xmm1, %%xmm4\n\t"
						"addps %%xmm4, %%xmm5\n\t"
						"movss 8(%[p_rhs_tmp], %%rcx), %%xmm4\n\t"
						"shufps $0, %%xmm4, %%xmm4\n\t"
						"mulps %%xmm2, %%xmm4\n\t"
						"addps %%xmm4, %%xmm5\n\t"
						"movss 12(%[p_rhs_tmp], %%rcx), %%xmm4\n\t"
						"shufps $0, %%xmm4, %%xmm4\n\t"
						"mulps %%xmm3, %%xmm4\n\t"
						"addps %%xmm4, %%xmm5\n\t"
						"movaps %%xmm5, (%[p_re_tmp], %%rcx)\n\t"
						"subq $16, %%rcx\n\t"
						"jge 1b\n\t"
						:
						: [p_lhs_tmp]"r"(p_lhs_tmp), [p_rhs_tmp]"r"(p_rhs_tmp), [p_re_tmp]"r"(p_re_tmp)
						: "rcx", "memory", "cc"
					);

					p_lhs_tmp += 4;
					p_rhs_tmp += 4;
				}
			}
		}

		benchmark::ClobberMemory();
	}

	 delete[] p_lhs;
	 delete[] p_rhs;
	 delete[] p_re;

	state.SetItemsProcessed(state.iterations() * M * K * N);
}
BENCHMARK(bm_t4n_mnk_prod_asm)
	->Args({ 16, 16, 16 })
	->Args({ 32, 32, 32 })
	->Args({ 128, 128, 128 })
	->Args({ 512, 512, 512 })
	->Args({ 1024, 1024, 1024 })
	->Args({ 2048, 2048, 2048 })
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
	->Args({ 8 * 8, 512, 32})
	->UseRealTime();

#endif

#define MATAZURE_SIMD
#ifdef MATAZURE_SIMD

#ifndef ANDROID

void bm_matrixf4x4_prod_vectorf40(benchmark::State &state) {
	auto n = state.range(0);
	point<simd_vector<float, 4>, 4> lhs;
	tensor<simd_vector<float, 4>, 1> ts_rhs(state.range(0));
	tensor<simd_vector<float, 4>, 1> ts_re(state.range(0));


	while (state.KeepRunning()) {
		for (int i = 0; i < ts_rhs.size(); ++i){
			ts_re[i] = puzzle::prod0(lhs, ts_rhs[i]);
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(state.iterations() * (16)* n);
}
BENCHMARK(bm_matrixf4x4_prod_vectorf40)
->Args({ 4 })
->Args({ 128 })
->Args({ 1024 })
->Args({ 1024 * 16 })
->Args({ 1024 * 64 })
->Args({ 1024 * 512 })
->Args({ 1024 * 1024 })
->Args({ 1024 * 1024 * 4})
->UseRealTime();

#endif

void bm_matrixf4x4_prod_vectorf41(benchmark::State &state) {
	auto n = state.range(0);
	point<simd_vector<float, 4>, 4> lhs;
	tensor<simd_vector<float, 4>, 1> ts_rhs(state.range(0));
	tensor<simd_vector<float, 4>, 1> ts_re(state.range(0));


	while (state.KeepRunning()) {
		for (int i = 0; i < ts_rhs.size(); ++i) {
			ts_re[i] = puzzle::prod1(lhs, ts_rhs[i]);
		}
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(state.iterations() * (16)* n);
}
BENCHMARK(bm_matrixf4x4_prod_vectorf41)
->Args({ 4 })
->Args({ 128 })
->Args({ 1024 })
->Args({ 1024 * 16 })
->Args({ 1024 * 64 })
->Args({ 1024 * 512 })
->Args({ 1024 * 1024 })
->Args({ 1024 * 1024 * 4 })
->UseRealTime();

#endif
