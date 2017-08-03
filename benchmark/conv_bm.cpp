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
BENCHMARK(bm_conv_sse2_expand)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

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
BENCHMARK(bm_conv_sse2_expand_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_sse2_inside_check_fixed(benchmark::State &state){
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

				if (MATAZURE_LIKELY(inside(pointi<2>{i, j}, pointi<2>{1,1}, ts_input.shape() - 2))){
					for_index(pointi<2>{0,0}, kenel.shape(), [&](pointi<2> kenel_idx){
 						sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} + kenel_idx -kenel_radius], kenel[kenel_idx]));
					});
				}else{
					for_index(pointi<2>{0,0}, kenel.shape(), [&](pointi<2> kenel_idx){
						__m128 value = _mm_setzero_ps();
						auto value_index = pointi<2>{i, j} + kenel_idx - kenel_radius;
						if (MATAZURE_LIKELY(inside(value_index, pointi<2>{1,1}, ts_input.shape() - 2))){
							value = ts_input[value_index];
						}
 						sum = _mm_add_ps(sum, _mm_mul_ps(value, kenel[kenel_idx]));
					});
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
BENCHMARK(bm_conv_sse2_inside_check_fixed)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_conv_sse2_4_caffe2(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));

	tensor<float, 2> ts_input0(ext);
	tensor<float, 2> ts_input1(ext);
	tensor<float, 2> ts_input2(ext);
	tensor<float, 2> ts_input3(ext);
	tensor<float, 2> ts_output0(ext);
	tensor<float, 2> ts_output1(ext);
	tensor<float, 2> ts_output2(ext);
	tensor<float, 2> ts_output3(ext);
	static_tensor<float, dim<3,3>> kenel0;
	static_tensor<float, dim<3,3>> kenel1;
	static_tensor<float, dim<3,3>> kenel2;
	static_tensor<float, dim<3,3>> kenel3;

	fill(ts_input0, 1.0f);
	fill(ts_input1, 2.0f);
	fill(ts_input2, 3.0f);
	fill(ts_input3, 4.0f);
	fill(ts_output0, 0.0f);
	fill(ts_output1, 0.0f);
	fill(ts_output2, 0.0f);
	fill(ts_output3, 0.0f);
	fill(kenel0, 1.0f);
	fill(kenel1, 1.0f);
	fill(kenel2, 1.0f);
	fill(kenel3, 1.0f);

	tensor<__m128, 2> ts_input(ext);
	tensor<__m128, 2> ts_output(ts_input.shape());
	static_tensor<__m128, dim<3,3>> kenel;

	auto width = ts_input.shape()[0];
	auto height = ts_input.shape()[1];
	const auto shape = ts_input.shape();

	while (state.KeepRunning()){
		for (int_t i = 0; i < ts_input.size(); ++i){
			ts_input[i] = _mm_set_ps(ts_input3[i], ts_input2[i], ts_input1[i], ts_input0[i]);
		}
		for (int_t i = 0; i < kenel.size(); ++i){
			kenel[i] = _mm_set_ps(kenel3[i], kenel2[i], kenel1[i], kenel0[i]);
		}

		auto kenel_radius = kenel.shape() / 2;

		for_index(pointi<2>{0,0}, ts_output.shape(), [&](pointi<2> idx){
				__m128 sum = _mm_setzero_ps();

				if (MATAZURE_LIKELY(inside(idx, pointi<2>{1,1}, ts_input.shape() - 1))){
					for_index(pointi<2>{0,0}, kenel.shape(), [&](pointi<2> kenel_idx){
 						sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[idx + kenel_idx -kenel_radius], kenel[kenel_idx]));
					});
				}else{
					for_index(pointi<2>{0,0}, kenel.shape(), [&](pointi<2> kenel_idx){
						__m128 value = _mm_setzero_ps();
						auto value_index = idx + kenel_idx - kenel_radius;
						if (MATAZURE_LIKELY(inside(value_index, pointi<2>{0,0}, ts_input.shape()))){
							value = ts_input[value_index];
						}
 						sum = _mm_add_ps(sum, _mm_mul_ps(value, kenel[kenel_idx]));
					});
				}

				ts_output[idx] = sum;
		});

		for (int_t i = 0; i < ts_input.size(); ++i){
		#ifndef __linux__
			ts_output0[i] = ts_output[i].m128_f32[0];
			ts_output1[i] = ts_output[i].m128_f32[1];
			ts_output2[i] = ts_output[i].m128_f32[2];
			ts_output3[i] = ts_output[i].m128_f32[3];
		#else
			ts_output0[i] = ts_output[i][0];
			ts_output1[i] = ts_output[i][1];
			ts_output2[i] = ts_output[i][2];
			ts_output3[i] = ts_output[i][3];
		#endif
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
BENCHMARK(bm_conv_sse2_4_caffe2)->Arg(7)->UseRealTime();

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void bm_img2col(benchmark::State &state){
	pointi<2> ext{};
	fill(ext, state.range(0));
	tensor<float, 2> mat_input(ext);
	tensor<float, 3> mat_re(pointi<3>{ext[0], ext[1], 9});
	fill(mat_input, 0.0f);
	fill(mat_re, 0.0f);

	static_tensor<float, dim<3,3>> sts_kernel;
	auto kernel_h = sts_kernel.shape()[1];
	auto kernel_w = sts_kernel.shape()[0];
	int dilation_h = 1;
	int dilation_w = 1;
	int pad_h = 1;
	int pad_w = 1;
	int output_w = mat_re.shape()[0];
	int output_h = mat_re.shape()[1];
	int height = mat_input.shape()[1];
	int width = mat_input.shape()[0];

	int stride_w = 1;
	int stride_h = 1;

	while (state.KeepRunning()){
		auto data_col = mat_re.data();
	 	auto data_im = mat_input.data();

		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (MATAZURE_UNLIKELY(!is_a_ge_zero_and_a_lt_b(input_row, height))) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					} else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (MATAZURE_LIKELY(is_a_ge_zero_and_a_lt_b(input_col, width))) {
								*(data_col++) = data_im[input_row * width + input_col];
							} else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}

	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * mat_re.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * mat_re.size());
}
BENCHMARK(bm_img2col)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_img2col_conv(benchmark::State &state){
	pointi<2> ext{};
	fill(ext, state.range(0));
	tensor<float, 2> mat_input(ext);
	tensor<float, 2> mat_re(ext);
	fill(mat_input, 1.1f);
	fill(mat_re, 1.1f);
	static_tensor<float, dim<3,3>> sts_kernel;
	fill(sts_kernel, 1.1f);

	auto kernel_h = sts_kernel.shape()[1];
	auto kernel_w = sts_kernel.shape()[0];
	int dilation_h = 1;
	int dilation_w = 1;
	int pad_h = 1;
	int pad_w = 1;
	int output_w = mat_re.shape()[0];
	int output_h = mat_re.shape()[1];
	int height = mat_input.shape()[1];
	int width = mat_input.shape()[0];

	int stride_w = 1;
	int stride_h = 1;

	while (state.KeepRunning()){
	 	auto data_im = mat_input.data();

		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				auto data_out = mat_re.data();
				auto weight = sts_kernel(kernel_col, kernel_row);
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (MATAZURE_UNLIKELY(!is_a_ge_zero_and_a_lt_b(input_row, height))) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_out++) += 0;
						}
					} else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (MATAZURE_LIKELY(is_a_ge_zero_and_a_lt_b(input_col, width))) {
								*(data_out++) += weight * data_im[input_row * width + input_col];
							} else {
								*(data_out++) += 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}

	#ifdef __linux__
		benchmark::ClobberMemory();
	#endif
	}

	state.SetBytesProcessed(state.iterations() * mat_re.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * mat_re.size() * 9);
}
BENCHMARK(bm_img2col_conv)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();
