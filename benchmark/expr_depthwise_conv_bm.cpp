#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <bm_config.hpp>
#include <matazure/tensor>
#include <matazure/expr/conv.hpp>
#include <emmintrin.h>
#include <immintrin.h>

using namespace matazure;

// #define __LOCAL_USE_OMP

void bm_conv_direct(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j) {
			for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i) {
				float sum = 0.0f;
				for (int_t n = 0; n < kenel.shape()[1]; ++n) {
					for (int_t m = 0; m < kenel.shape()[0]; ++m) {
						sum += ts_input[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * kenel[pointi<2>{m, n}];
					}
				}

				ts_output[pointi<2>{i, }] = sum;
			}
		}
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_conv_direct)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

// void bm_conv_general(benchmark::State& state){
// 	pointi<2> ext;
// 	fill(ext, state.range(0));
// 	tensor<float, 2> ts_input(ext);
// 	tensor<float, 2> ts_output(ts_input.shape());
// 	static_tensor<float, dim<3,3>> kenel;
// 	fill(kenel, 1.0f);
//
// 	while (state.KeepRunning()){
// 		copy(puzzle::conv_general(ts_input, kenel), ts_output);
// 	}
//
// 	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
// 	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
// }
// BENCHMARK(bm_conv_general)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_expand(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for(int_t j = 1; j < ts_input.shape()[1] - 1; ++j) {
			for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i) {
				float sum = 0.0f;
				sum += ts_input[pointi<2>{i, j} +pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
				sum += ts_input[pointi<2>{i, j} +pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];

				ts_output[pointi<2>{i, j}] = sum;
			}
		}
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_conv_expand)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_conv_kenel3x3(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		expr::conv_kenel3x3(ts_input, kenel, ts_output);
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_conv_conv_kenel3x3)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_inside_check(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;

		auto width = ts_input.shape()[0];
		auto height = ts_input.shape()[1];
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for(int_t j = 0; j < ts_input.shape()[1]; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0]; ++i) {
				if (i > 0 && j > 0 && i < width && j < height){
					float sum = 0.0f;
					for (int_t n = 0; n < kenel.shape()[1]; ++n) {
						for (int_t m = 0; m < kenel.shape()[0]; ++m) {
							sum += ts_input[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * kenel[pointi<2>{m, n}];
						}
					}
					ts_output[pointi<2>{i, }] = sum;
				}else{
					ts_output[pointi<2>{i, j}] = 0.0f;
				}
			}
		}
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_conv_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_expand_inside_check(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;

		auto width = ts_input.shape()[0];
		auto height = ts_input.shape()[1];
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for(int_t j = 0; j < ts_input.shape()[1]; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0]; ++i) {
				if (i > 0 && j > 0 && i < width && j < height){
					float sum = 0.0f;
					sum += ts_input[pointi<2>{i, j} +pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
					sum += ts_input[pointi<2>{i, j} +pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];

					ts_output[pointi<2>{i, j}] = sum;
				}else{
					ts_output[pointi<2>{i, j}] = 0.0f;
				}
			}
		}
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_conv_expand_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();
// BENCHMARK(bm_conv_expand_inside_check)->RangeMultiplier(2)->Range(16, 256)->UseRealTime();

void bm_conv_inside_check_block(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;

		auto width = ts_input.shape()[0];
		auto height = ts_input.shape()[1];
		int_t block_size = 16;

	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for(int_t j = 0; j < ts_input.shape()[1] / block_size; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0] / block_size; ++i) {
				for (int_t m = 0; m < block_size; ++m){
					for (int_t n = 0; n < block_size; ++n){
						auto x = i * block_size + m;
						auto y = j * block_size + n;
						if (x > 0 && y > 0 && x < width && y < height){
							float sum = 0.0f;
							for (int_t n = 0; n < kenel.shape()[1]; ++n) {
								for (int_t m = 0; m < kenel.shape()[0]; ++m) {
									sum += ts_input[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * kenel[pointi<2>{m, n}];
								}
							}

							ts_output[pointi<2>{i, }] = sum;
						}else{
							ts_output[pointi<2>{x, y}] = 0.0f;
						}
					}
				}
			}
		}
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_conv_inside_check_block)->RangeMultiplier(2)->Range(32, 256)->UseRealTime();

void bm_conv_inside_check_block_with_block_tensor(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));

	typedef dim<5,5> block_type;
	auto block_dim = block_type::value();
	block_tensor<float, block_type> ts_src(ext / block_dim);
	block_tensor<float, block_type> ts_dst(ext / block_dim);

	auto ts_input = global_view(ts_src);
	auto ts_output = global_view(ts_dst);

	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;

		auto width = ts_input.shape()[0];
		auto height = ts_input.shape()[1];

		int_t block_size = block_dim[0];

		#ifdef __LOCAL_USE_OMP
			#pragma omp parallel for collapse(2)
		#endif
		for(int_t j = 0; j < ts_input.shape()[1] / block_size; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0] / block_size; ++i) {
				for (int_t m = 0; m < block_size; ++m){
					for (int_t n = 0; n < block_size; ++n){
						auto x = i * block_size + m;
						auto y = j * block_size + n;
						if (x > 0 && y > 0 && x < width && y < height){
							float sum = 0.0f;
							for (int_t n = 0; n < kenel.shape()[1]; ++n) {
								for (int_t m = 0; m < kenel.shape()[0]; ++m) {
									sum += ts_input[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * kenel[pointi<2>{m, n}];
								}
							}

							ts_output[pointi<2>{i, }] = sum;
						}else{
							ts_output[pointi<2>{x, y}] = 0.0f;
						}
					}
				}
			}
		}
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_conv_inside_check_block_with_block_tensor)->RangeMultiplier(2)->Range(32, 256)->UseRealTime();

void bm_conv_outside_check(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel{};
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j) {
			for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i) {
				float sum = 0.0f;
				for (int_t n = 0; n < kenel.shape()[1]; ++n) {
					for (int_t m = 0; m < kenel.shape()[0]; ++m) {
						sum += ts_input[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * kenel[pointi<2>{m, n}];
					}
				}

				ts_output[pointi<2>{i, }] = sum;
			}
		}

		auto last_row_pos = ts_input.shape()[1] - 1;
		auto last_col_pos = ts_input.shape()[0] - 1;

		//left top corner
		{
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, 0}] = sum;
		}

		// top right corner
		{
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{last_row_pos, 0}] = sum;
		}

		//left  bottom
		{
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, last_row_pos}] = sum;

		}

		//right bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{last_col_pos, last_row_pos}] = sum;
		}

		//top
		for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i){
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{i, 0} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			// sum += ts_input[pointi<2>{i, 0} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{i, 0} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{i, 0}] = sum;
		}

		// //left
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j){
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{0, j} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			// sum += ts_input[pointi<2>{0, j} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{0, j} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
		}

		//right
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j){
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			// sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{last_col_pos, j}] = sum;
		}

		//bottom
		for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i){
			float sum = 0.0f;
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			// sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{i, last_row_pos}] = sum;
		}
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_conv_outside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

// void bm_conv_halo(benchmark::State &state){
// 	pointi<2> ext;
// 	fill(ext, state.range(0));
// 	tensor<float, 2> ts_input(ext);
// 	tensor<float, 2> ts_output(ts_input.shape());
// 	tensor<float, 2> ts_halo(ext+1);
// 	static_tensor<float, dim<3,3>> kenel{};
// 	fill(ts_input, 1.0f);
// 	fill(kenel, 1.0f);
//
// 	while (state.KeepRunning()){
// 		auto lts_valid = section(ts_halo, pointi<2>{1, 1}, ts_input.shape());
// 		copy(ts_input, lts_valid);
// 		auto top0 = slice<0>(ts_halo, 0);
// 		auto top1 = slice<0>(ts_halo, 1);
// 		copy(top1, top0);
// 		auto bottom0 = slice<0>(ts_halo, ts_halo.shape()[1] - 1);
// 		auto bottom1 = slice<0>(ts_halo, ts_halo.shape()[1] - 2);
// 		copy(bottom1, bottom0);
// 		auto left0 = slice<1>(ts_halo, 0);
// 		auto left1 = slice<1>(ts_halo, 1);
// 		auto right0 = slice<1>(ts_halo, ts_halo.shape()[0] - 1);
// 		auto right1 = slice<1>(ts_halo, ts_halo.shape()[0] - 2);
//
// 		auto kenel_radius = kenel.shape() / 2;
// 	#ifdef __LOCAL_USE_OMP
// 		#pragma omp parallel for collapse(2)
// 	#endif
// 		for (int_t j = 1; j < ts_halo.shape()[1] - 1; ++j) {
// 			for (int_t i = 1; i < ts_halo.shape()[0] - 1; ++i) {
// 				float sum = 0.0f;
// 				for (int_t n = 0; n < kenel.shape()[1]; ++n) {
// 					for (int_t m = 0; m < kenel.shape()[0]; ++m) {
// 						sum += ts_input[pointi<2>{i, j} +pointi<2>{m, n} -kenel_radius] * kenel[pointi<2>{m, n}];
// 					}
// 				}
//
// 				ts_output[pointi<2>{i, }] = sum;
// 			}
// 		}
// 	}
//
// 	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
// 	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
// }
// BENCHMARK(bm_conv_halo)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_linear(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(kenel, 1.0f);

	pointi<9> poses;
	poses[0] = -1-ts_input.shape()[0];
	poses[1] = -ts_input.shape()[0];
	poses[2] = 1-ts_input.shape()[0];
	poses[3] = -1;
	poses[4] =  0;
	poses[5] = 1;
	poses[6] = -1+ts_input.shape()[0];
	poses[7] = ts_input.shape()[0];
	poses[8] = 1+ts_input.shape()[0];

	pointf<9> weights;
	fill(weights, 1.0f);

	while (state.KeepRunning()){
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for
	#endif
		for (int_t i = 0; i < ts_input.size(); ++i){
			float sum = 0.0f;
			for (int_t j = 0; j < poses.size(); ++j){
				sum += ts_input[i+poses[j]] * weights[j];
			}
			ts_output[i] = sum;
		}

		auto last_row_pos = ts_input.shape()[1] - 1;
		auto last_col_pos = ts_input.shape()[0] - 1;

		auto kenel_radius = kenel.shape() / 2;
		//left top corner
		{
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{0, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, 0}] = sum;
		}

		// top right corner
		{
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{last_row_pos, 0}] = sum;
		}

		//left  bottom
		{
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, last_row_pos}] = sum;

		}

		//right bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{last_col_pos, last_row_pos}] = sum;
		}

		//top
		for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i){
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{i, 0} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			// sum += ts_input[pointi<2>{i, 0} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{i, 0} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{i, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{i, 0}] = sum;
		}

		// //left
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j){
			float sum = 0.0f;
			// sum += ts_input[pointi<2>{0, j} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			// sum += ts_input[pointi<2>{0, j} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{0, j} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, j}] = sum;
		}

		//right
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j){
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			// sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			// sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{last_col_pos, j}] = sum;
		}

		//bottom
		for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i){
			float sum = 0.0f;
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			// sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			// sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			// sum += ts_input[pointi<2>{i, last_row_pos} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{i, last_row_pos}] = sum;
		}
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_conv_linear)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_sse2_expand(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<__m128, 2> ts_input(ext);
	tensor<__m128, 2> ts_output(ts_input.shape());
	static_tensor<__m128, dim<3,3>> kenel;
	for_each(ts_input, [](__m128 &e){
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});
	for_each(ts_output, [](__m128 &e){
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});
	for_each(kenel, [](__m128 &e){
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for(int_t j = 1; j < ts_input.shape()[1] - 1; ++j) {
			for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i) {
				__m128 sum = _mm_setzero_ps();

				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 0} -kenel_radius], kenel[pointi<2>{0, 0}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 0} -kenel_radius], kenel[pointi<2>{1, 0}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 0} -kenel_radius], kenel[pointi<2>{2, 0}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 1} -kenel_radius], kenel[pointi<2>{0, 1}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 1} -kenel_radius], kenel[pointi<2>{1, 1}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 1} -kenel_radius], kenel[pointi<2>{2, 1}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 2} -kenel_radius], kenel[pointi<2>{0, 2}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 2} -kenel_radius], kenel[pointi<2>{1, 2}]));
				sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 2} -kenel_radius], kenel[pointi<2>{2, 2}]));

				ts_output[pointi<2>{i, j}] = sum;
			}
		}

	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(__m128));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size() * sizeof(__m128) / sizeof(float));
}
BENCHMARK(bm_conv_sse2_expand)->RangeMultiplier(2)->Range(16, 128)->UseRealTime();

void bm_conv_sse2_expand_inside_check(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<__m128, 2> ts_input(ext);
	tensor<__m128, 2> ts_output(ts_input.shape());
	static_tensor<__m128, dim<3,3>> kenel;
	for_each(ts_input, [](__m128 &e){
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});
	for_each(ts_output, [](__m128 &e){
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});
	for_each(kenel, [](__m128 &e){
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});

	auto width = ts_input.shape()[0];
	auto height = ts_input.shape()[1];
	const auto shape = ts_input.shape();

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
	#ifdef __LOCAL_USE_OMP
		#pragma omp parallel for collapse(2)
	#endif
		for(int_t j = 0; j < ts_input.shape()[1] ; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0] ; ++i) {

				__m128 sum = _mm_setzero_ps();

				// if (i > 1 && j > 1 && i < width - 1 && i < height - 1){
				if (MATAZURE_LIKELY(inside(pointi<2>{i, j}, pointi<2>{1,1}, ts_input.shape()))){
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 0} -kenel_radius], kenel[pointi<2>{0, 0}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 0} -kenel_radius], kenel[pointi<2>{1, 0}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 0} -kenel_radius], kenel[pointi<2>{2, 0}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 1} -kenel_radius], kenel[pointi<2>{0, 1}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 1} -kenel_radius], kenel[pointi<2>{1, 1}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 1} -kenel_radius], kenel[pointi<2>{2, 1}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 2} -kenel_radius], kenel[pointi<2>{0, 2}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 2} -kenel_radius], kenel[pointi<2>{1, 2}]));
					sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 2} -kenel_radius], kenel[pointi<2>{2, 2}]));
				}

				ts_output[pointi<2>{i, j}] = sum;
			}
		}

	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(__m128));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size() * sizeof(__m128) / sizeof(float));
}
BENCHMARK(bm_conv_sse2_expand_inside_check)->RangeMultiplier(2)->Range(16, 128)->UseRealTime();

// void bm_conv_sperate_sse2(benchmark::State &state){
// 	pointi<2> ext;
// 	fill(ext, state.range(0));
// 	tensor<float, 2> ts_input(ext);
// 	tensor<float, 2> ts_output(ts_input.shape());
// 	static_tensor<float, dim<3,3>> kenel;
// 	fill(ts_input, 1.0f);
// 	fill(kenel, 1.0f);
//
// 	while (state.KeepRunning()){
// 		auto kenel_radius = kenel.shape() / 2;
// 	#ifdef __LOCAL_USE_OMP
// 		#pragma omp parallel for collapse(2)
// 	#endif
// 		for (int_t j = 0; j < ts_input.shape()[1]; ++j) {
// 			for (int_t i = 4; i < ts_input.shape()[0] - 4; i += 4) {
// 				__m128 sum = _mm_setzero_ps();
//
// 				__m128 pixel00 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 0} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{0, 0} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{0, 0} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{0, 0} -kenel_radius]);
// 				__m128 pixel10 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 0} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{1, 0} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{1, 0} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{1, 0} -kenel_radius]);
// 				__m128 pixel20 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 0} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{2, 0} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{2, 0} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{2, 0} -kenel_radius]);
// 				__m128 pixel01 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 1} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{0, 1} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{0, 1} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{0, 1} -kenel_radius]);
// 				__m128 pixel11 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 1} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{1, 1} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{1, 1} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{1, 1} -kenel_radius]);
// 				__m128 pixel21 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 1} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{2, 1} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{2, 1} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{2, 1} -kenel_radius]);
// 				__m128 pixel02 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 2} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{0, 2} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{0, 2} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{0, 2} -kenel_radius]);
// 				__m128 pixel12 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 2} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{1, 2} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{1, 2} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{1, 2} -kenel_radius]);
// 				__m128 pixel22 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 2} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{2, 2} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{2, 2} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{2, 2} -kenel_radius]);
//
// 				auto weight00 = _mm_set_ps1(kenel[pointi<2>{0, 0}]);
// 				auto weight10 = _mm_set_ps1(kenel[pointi<2>{1, 0}]);
// 				auto weight20 = _mm_set_ps1(kenel[pointi<2>{2, 0}]);
// 				auto weight01 = _mm_set_ps1(kenel[pointi<2>{0, 1}]);
// 				auto weight11 = _mm_set_ps1(kenel[pointi<2>{1, 1}]);
// 				auto weight21 = _mm_set_ps1(kenel[pointi<2>{2, 1}]);
// 				auto weight02 = _mm_set_ps1(kenel[pointi<2>{0, 2}]);
// 				auto weight12 = _mm_set_ps1(kenel[pointi<2>{1, 2}]);
// 				auto weight22 = _mm_set_ps1(kenel[pointi<2>{2, 2}]);
//
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel00, weight00));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel10, weight10));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel20, weight20));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel01, weight01));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel11, weight11));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel21, weight21));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel02, weight02));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel12, weight12));
// 				sum = _mm_add_ps(sum, _mm_mul_ps(pixel22, weight22));
//
// 				auto dst_p = ts_output.data() + index2offset(pointi<2>{i, j}, ts_output.stride(), first_major_t{});
// 				_mm_store_ps(dst_p, sum);
// 			}
// 		}
// 	}
//
// 	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
// 	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
// }
// BENCHMARK(bm_conv_sperate_sse2)->RangeMultiplier(2)->Range(16, 256)->UseRealTime();
