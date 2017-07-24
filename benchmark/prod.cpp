#include <benchmark/benchmark.h>
#include <bm_config.hpp>

#include <matazure/tensor>
#include <emmintrin.h>
#include <immintrin.h>

#include <Eigen/Core>
#include <Eigen/Dense>

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
					re += p_lhs[k + m * K] * p_rhs[k + m * K];
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
BENCHMARK(bm_prod_simple_gold)->Args({2048, 16, 2048})->UseRealTime();

void bm_eigen_prod(benchmark::State &state){
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_lhs(state.range(0), state.range(1));
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_rhs(state.range(1), state.range(2));
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mat_output(state.range(0), state.range(2));
	mat_lhs.setOnes();
	mat_rhs.setOnes();

	while (state.KeepRunning()){
		mat_output = mat_lhs * mat_rhs;
	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetItemsProcessed(state.iterations() * mat_lhs.rows() * mat_lhs.cols() * mat_rhs.cols());
}
BENCHMARK(bm_eigen_prod)->Args({2048, 16, 2048 })->UseRealTime();

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
BENCHMARK(bm_prod)->Args({2048, 16, 2048})->UseRealTime();
