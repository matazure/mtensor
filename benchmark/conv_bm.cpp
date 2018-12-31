#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

using namespace matazure;

void bm_gold_conv_3x3(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
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

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_inside_check(benchmark::State &state){
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

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 2;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_outside_check(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel{};
	fill(ts_input, 1.0f);
	fill(kenel, 1.0f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
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
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, 0}] = sum;
		}

		// top right corner
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			ts_output[pointi<2>{last_row_pos, 0}] = sum;
		}

		//left  bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			ts_output[pointi<2>{0, last_row_pos}] = sum;

		}

		//right bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			ts_output[pointi<2>{last_col_pos, last_row_pos}] = sum;
		}

		//top
		for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i){
			float sum = 0.0f;
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
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
		}

		//right
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j){
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
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
			ts_output[pointi<2>{i, last_row_pos}] = sum;
		}

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_outside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_expand(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3,3>> kenel;
	fill(ts_input, 1.0f);
	fill(kenel, 1.1f);

	while (state.KeepRunning()){
		auto kenel_radius = kenel.shape() / 2;
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

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_expand)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_expand_inside_check(benchmark::State &state){
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

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_expand_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_expand_outside_check(benchmark::State &state){
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
		for(int_t j = 1; j < ts_input.shape()[1]; ++j) {
			for (int_t i = 1; i < ts_input.shape()[0]; ++i) {
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

		auto last_row_pos = ts_input.shape()[1] - 1;
		auto last_col_pos = ts_input.shape()[0] - 1;
		//left top corner
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, 0}] = sum;
		}

		// top right corner
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			ts_output[pointi<2>{last_row_pos, 0}] = sum;
		}

		//left  bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			ts_output[pointi<2>{0, last_row_pos}] = sum;

		}

		//right bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			ts_output[pointi<2>{last_col_pos, last_row_pos}] = sum;
		}

		//top
		for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i){
			float sum = 0.0f;
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
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
		}

		//right
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j){
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
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
			ts_output[pointi<2>{i, last_row_pos}] = sum;
		}

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() *  ts_output.size() * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_expand_outside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_linear(benchmark::State &state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3, 3>> kenel;
	fill(kenel, 1.0f);

	pointi<9> poses;
	poses[0] = -1 - ts_input.shape()[0];
	poses[1] = -ts_input.shape()[0];
	poses[2] = 1 - ts_input.shape()[0];
	poses[3] = -1;
	poses[4] = 0;
	poses[5] = 1;
	poses[6] = -1 + ts_input.shape()[0];
	poses[7] = ts_input.shape()[0];
	poses[8] = 1 + ts_input.shape()[0];

	pointf<9> weights;
	fill(weights, 1.0f);

	while (state.KeepRunning()) {
		for (int_t i = 0; i < ts_input.size(); ++i) {
			float sum = 0.0f;
			for (int_t j = 0; j < poses.size(); ++j) {
				sum += ts_input[i + poses[j]] * weights[j];
			}
			ts_output[i] = sum;
		}

		auto last_row_pos = ts_input.shape()[1] - 1;
		auto last_col_pos = ts_input.shape()[0] - 1;

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_linear)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_linear_outside_check(benchmark::State &state){
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
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, 0} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, 0}] = sum;
		}

		// top right corner
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_row_pos, 0} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			ts_output[pointi<2>{last_row_pos, 0}] = sum;
		}

		//left  bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, last_row_pos} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			ts_output[pointi<2>{0, last_row_pos}] = sum;

		}

		//right bottom
		{
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, last_row_pos} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			ts_output[pointi<2>{last_col_pos, last_row_pos}] = sum;
		}

		//top
		for (int_t i = 1; i < ts_input.shape()[0] - 1; ++i){
			float sum = 0.0f;
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
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 0} -kenel_radius] * kenel[pointi<2>{2, 0}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 1} -kenel_radius] * kenel[pointi<2>{2, 1}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
			sum += ts_input[pointi<2>{0, j} + pointi<2>{2, 2} -kenel_radius] * kenel[pointi<2>{2, 2}];
			ts_output[pointi<2>{0, j}] = sum;
		}

		//right
		for (int_t j = 1; j < ts_input.shape()[1] - 1; ++j){
			float sum = 0.0f;
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 0} -kenel_radius] * kenel[pointi<2>{0, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 0} -kenel_radius] * kenel[pointi<2>{1, 0}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 1} -kenel_radius] * kenel[pointi<2>{0, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 1} -kenel_radius] * kenel[pointi<2>{1, 1}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{0, 2} -kenel_radius] * kenel[pointi<2>{0, 2}];
			sum += ts_input[pointi<2>{last_col_pos, j} + pointi<2>{1, 2} -kenel_radius] * kenel[pointi<2>{1, 2}];
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
			ts_output[pointi<2>{i, last_row_pos}] = sum;
		}

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_linear_outside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_linear_inside_check(benchmark::State &state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tensor<float, 2> ts_output(ts_input.shape());
	static_tensor<float, dim<3, 3>> kenel;
	fill(kenel, 1.0f);

	pointi<9> poses;
	poses[0] = -1 - ts_input.shape()[0];
	poses[1] = -ts_input.shape()[0];
	poses[2] = 1 - ts_input.shape()[0];
	poses[3] = -1;
	poses[4] = 0;
	poses[5] = 1;
	poses[6] = -1 + ts_input.shape()[0];
	poses[7] = ts_input.shape()[0];
	poses[8] = 1 + ts_input.shape()[0];

	pointf<9> weights;
	fill(weights, 1.0f);

	while (state.KeepRunning()) {
		int_t i = 0;
		auto width = ts_input.shape()[0];
		auto height = ts_input.shape()[1];
		for (int_t n = 0; n < ts_input.shape()[1]; ++n) {
			for (int_t m = 0; m < ts_input.shape()[0]; ++m, ++i) {
				if (m > 0 && n > 0 && m < width && n < height) {
					float sum = 0.0f;
					for (int_t j = 0; j < poses.size(); ++j) {
						sum += ts_input[i + poses[j]] * weights[j];
					}
					ts_output[i] = sum;
				}
				else {
					ts_output[i] = 0.0f;
				}
			}
		}

		auto last_row_pos = ts_input.shape()[1] - 1;
		auto last_col_pos = ts_input.shape()[0] - 1;

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
BENCHMARK(bm_gold_conv_3x3_linear_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void bm_gold_img2col(benchmark::State &state){
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

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * mat_re.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * mat_re.size());
}
BENCHMARK(bm_gold_img2col)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_img2col(benchmark::State &state){
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

		benchmark::ClobberMemory();
	}

	state.SetBytesProcessed(state.iterations() * mat_re.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * mat_re.size() * 9);
}
BENCHMARK(bm_gold_conv_3x3_img2col)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

#ifdef MATAZURE_SSE

void bm_gold_conv_3x3_sse2_expand(benchmark::State &state){
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

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(__m128));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size() * sizeof(__m128) / sizeof(float));
}
BENCHMARK(bm_gold_conv_3x3_sse2_expand)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_sse2_expand_inside_check(benchmark::State &state){
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
		for(int_t j = 0; j < ts_input.shape()[1] ; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0] ; ++i) {

				__m128 sum = _mm_setzero_ps();

				if (MATAZURE_LIKELY(inside_range(pointi<2>{i, j}, pointi<2>{1,1}, ts_input.shape()))){
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

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(__m128));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size() * sizeof(__m128) / sizeof(float));
}
BENCHMARK(bm_gold_conv_3x3_sse2_expand_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_sse2_inside_check(benchmark::State &state){
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
		for(int_t j = 0; j < ts_input.shape()[1] ; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0] ; ++i) {

				__m128 sum = _mm_setzero_ps();;

				if (MATAZURE_LIKELY(inside_range(pointi<2>{i, j}, pointi<2>{1,1}, ts_input.shape() - 2))){
					for_index(pointi<2>{0,0}, kenel.shape(), [&](pointi<2> kenel_idx){
 						sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} + kenel_idx -kenel_radius], kenel[kenel_idx]));
					});
				}else{
					for_index(pointi<2>{0,0}, kenel.shape(), [&](pointi<2> kenel_idx){
						__m128 value = _mm_setzero_ps();
						auto value_index = pointi<2>{i, j} + kenel_idx - kenel_radius;
						if (MATAZURE_LIKELY(inside_range(value_index, pointi<2>{1,1}, ts_input.shape() - 2))){
							value = ts_input[value_index];
						}
 						sum = _mm_add_ps(sum, _mm_mul_ps(value, kenel[kenel_idx]));
					});
				}

				ts_output[pointi<2>{i, j}] = sum;
			}
		}

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(__m128));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size() * sizeof(__m128) / sizeof(float));
}
BENCHMARK(bm_gold_conv_3x3_sse2_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_sse2_op_inside_check(benchmark::State &state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<__m128, 2> ts_input(ext);
	tensor<__m128, 2> ts_output(ts_input.shape());
	static_tensor<__m128, dim<3, 3>> kenel;
	for_each(ts_input, [](__m128 &e) {
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});
	for_each(ts_output, [](__m128 &e) {
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});
	for_each(kenel, [](__m128 &e) {
		e = _mm_set_ps(1.1f, 1.2f, 1.2f, 1.3f);
	});

	auto width = ts_input.shape()[0];
	auto height = ts_input.shape()[1];
	const auto shape = ts_input.shape();

	while (state.KeepRunning()) {
		auto kenel_radius = kenel.shape() / 2;
		for (int_t j = 0; j < ts_input.shape()[1]; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0]; ++i) {

				__m128 sum = _mm_setzero_ps();;

				if (MATAZURE_LIKELY(inside_range(pointi<2>{i, j}, pointi<2>{1, 1}, ts_input.shape() - 2))) {
					for_index(pointi<2>{0, 0}, kenel.shape(), [&](pointi<2> kenel_idx) {
						sum = _mm_add_ps(sum, _mm_mul_ps(ts_input[pointi<2>{i, j} +kenel_idx - kenel_radius], kenel[kenel_idx]));
					});
				}
				else {
					for_index(pointi<2>{0, 0}, kenel.shape(), [&](pointi<2> kenel_idx) {
						__m128 value = _mm_setzero_ps();
						auto value_index = pointi<2>{ i, j } +kenel_idx - kenel_radius;
						if (MATAZURE_LIKELY(inside_range(value_index, pointi<2>{1, 1}, ts_input.shape() - 2))) {
							value = ts_input[value_index];
						}
						sum = value * kenel[kenel_idx];
					});
				}

				ts_output[pointi<2>{i, j}] = sum;
			}
		}

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y) { return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(__m128));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size() * sizeof(__m128) / sizeof(float));
}
BENCHMARK(bm_gold_conv_3x3_sse2_op_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

#endif

#ifdef MATAZURE_NEON

void bm_gold_conv_3x3_neon2_op_inside_check(benchmark::State &state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<neon_vector<float, 4>, 2> ts_input(ext);
	tensor<neon_vector<float, 4>, 2> ts_output(ts_input.shape());
	static_tensor<neon_vector<float, 4>, dim<3, 3>> kenel;

	for_each(ts_input, [](neon_vector<float, 4> &e) {
		point<float, 4> tmp = { 1.1f, 1.2f, 1.2f, 1.3f };
		e = { vld1q_f32(reinterpret_cast<float *>(&tmp)) };

	});
	for_each(ts_output, [](neon_vector<float, 4> &e) {
		point<float, 4> tmp = { 1.1f, 1.2f, 1.2f, 1.3f };
		e = { vld1q_f32(reinterpret_cast<float *>(&tmp)) };

	});
	for_each(kenel, [](neon_vector<float, 4> &e) {
		point<float, 4> tmp = { 1.1f, 1.2f, 1.2f, 1.3f };
		e = { vld1q_f32(reinterpret_cast<float *>(&tmp)) };
	});

	auto width = ts_input.shape()[0];
	auto height = ts_input.shape()[1];
	const auto shape = ts_input.shape();

	while (state.KeepRunning()) {
		auto kenel_radius = kenel.shape() / 2;
		for (int_t j = 0; j < ts_input.shape()[1]; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0]; ++i) {

				neon_vector<float, 4> sum = zero<neon_vector<float, 4>>::value();

				if (MATAZURE_LIKELY(inside_range(pointi<2>{i, j}, pointi<2>{1, 1}, ts_input.shape() - 2))) {
					for_index(pointi<2>{0, 0}, kenel.shape(), [&](pointi<2> kenel_idx) {
						sum = sum + (ts_input[pointi<2>{i, j} +kenel_idx - kenel_radius] * kenel[kenel_idx]);
					});
				} else {
					for_index(pointi<2>{0, 0}, kenel.shape(), [&](pointi<2> kenel_idx) {
						neon_vector<float, 4> value = zero<neon_vector<float, 4>>::value();
						auto value_index = pointi<2>{ i, j } +kenel_idx - kenel_radius;
						if (MATAZURE_LIKELY(inside_range(value_index, pointi<2>{1, 1}, ts_input.shape() - 2))) {
							value = ts_input[value_index];
						}
						sum = value * kenel[kenel_idx];
					});
				}

				ts_output[pointi<2>{i, j}] = sum;
			}
		}

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y) { return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * 4 * 4);
	state.SetItemsProcessed(state.iterations() * valid_size * 9 * 4);
}
BENCHMARK(bm_gold_conv_3x3_neon2_op_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

void bm_gold_conv_3x3_neon_float16_op_inside_check(benchmark::State &state) {
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<neon_vector<float16_t, 8>, 2> ts_input(ext);
	tensor<neon_vector<float16_t, 8>, 2> ts_output(ts_input.shape());
	static_tensor<neon_vector<float16_t, 8>, dim<3, 3>> kenel;
	for_each(ts_input, [](neon_vector<float16_t, 8> &e) {
		point<float16_t, 8> tmp = { 1.1f, 1.2f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f };
		e = { vld1q_f16(reinterpret_cast<float16_t *>(&tmp)) };

	});
	for_each(ts_output, [](neon_vector<float16_t, 8> &e) {
		point<float16_t, 8> tmp = { 1.1f, 1.2f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f };
		e = { vld1q_f16(reinterpret_cast<float16_t *>(&tmp)) };

	});
	for_each(kenel, [](neon_vector<float16_t, 8> &e) {
		point<float16_t, 8> tmp = { 1.1f, 1.2f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f };
		e = { vld1q_f16(reinterpret_cast<float16_t *>(&tmp)) };
	});

	auto width = ts_input.shape()[0];
	auto height = ts_input.shape()[1];
	const auto shape = ts_input.shape();

	while (state.KeepRunning()) {
		auto kenel_radius = kenel.shape() / 2;
		for (int_t j = 0; j < ts_input.shape()[1]; ++j) {
			for (int_t i = 0; i < ts_input.shape()[0]; ++i) {

				neon_vector<float16_t, 8> sum = zero<neon_vector<float16_t, 8>>::value();

				if (MATAZURE_LIKELY(inside_range(pointi<2>{i, j}, pointi<2>{1, 1}, ts_input.shape() - 2))) {
					for_index(pointi<2>{0, 0}, kenel.shape(), [&](pointi<2> kenel_idx) {
						sum = sum + (ts_input[pointi<2>{i, j} +kenel_idx - kenel_radius] * kenel[kenel_idx]);
					});
				} else {
					for_index(pointi<2>{0, 0}, kenel.shape(), [&](pointi<2> kenel_idx) {
						neon_vector<float16_t, 8> value = zero<neon_vector<float16_t, 8>>::value();
						auto value_index = pointi<2>{ i, j } +kenel_idx - kenel_radius;
						if (MATAZURE_LIKELY(inside_range(value_index, pointi<2>{1, 1}, ts_input.shape() - 2))) {
							value = ts_input[value_index];
						}
						sum = value * kenel[kenel_idx];
					});
				}

				ts_output[pointi<2>{i, j}] = sum;
			}
		}

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape() - 1;
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y) { return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * 4 * 8);
	state.SetItemsProcessed(state.iterations() * valid_size * 9 * 8);
}
BENCHMARK(bm_gold_conv_3x3_neon_float16_op_inside_check)->Arg(7)->Arg(14)->Arg(28)->Arg(56)->Arg(112)->Arg(224)->UseRealTime();

#endif

template <typename _ValueType>
void bm_conv_lazy_array_index_unclamp_3x3(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<_ValueType, 2> ts_input(ext);
	tensor<_ValueType, 2> ts_output(ts_input.shape() - pointi<2>::all(2));
	static_tensor<_ValueType, dim<3,3>> kenel;

	while (state.KeepRunning()){
		auto ts_tmp = section(puzzle::conv_lazy_array_index_unclamp(ts_input, kenel), pointi<2>::all(1), ts_input.shape() - pointi<2>::all(2));
		copy(ts_tmp, ts_output);

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape();
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(_ValueType));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
#define BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(ValueType) \
auto bm_tensor_##ValueType##_rank2_conv_layz_array_index_unclamp_kernel3x3 = bm_conv_lazy_array_index_unclamp_3x3<ValueType>; \
BENCHMARK(bm_tensor_##ValueType##_rank2_conv_layz_array_index_unclamp_kernel3x3)->RangeMultiplier(bm_config::range_multiplier<ValueType, 2, host_tag>())->Range(bm_config::min_shape<ValueType, 2, host_tag>(), bm_config::max_shape<ValueType, 2, host_tag>())->UseRealTime();

BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(byte)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(int16_t)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(int32_t)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(int64_t)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(float)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(double)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(point4f)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_UNCLAMP_KERNEL3x3(hete_float32x4_t)

template <typename _ValueType>
void conv_lazy_array_index_inside_clamp_zero(benchmark::State &state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<_ValueType, 2> ts_input(ext);
	tensor<_ValueType, 2> ts_output(ts_input.shape());
	static_tensor<_ValueType, dim<3,3>> kenel;

	while (state.KeepRunning()){
		copy(puzzle::conv_lazy_array_index_inside_clamp_zero(ts_input, kenel), ts_output);

		benchmark::ClobberMemory();
	}

	auto valid_shape = ts_output.shape();
	auto valid_size = reduce(valid_shape, 1, [](int_t x, int_t y){ return x * y; });
	state.SetBytesProcessed(state.iterations() * valid_size * sizeof(_ValueType));
	state.SetItemsProcessed(state.iterations() * valid_size * kenel.size());
}
#define BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(ValueType) \
auto bm_tensor_##ValueType##_rank2_conv_lazy_array_index_inside_clamp_zero_kernel3x3 = conv_lazy_array_index_inside_clamp_zero<ValueType>; \
BENCHMARK(bm_tensor_##ValueType##_rank2_conv_lazy_array_index_inside_clamp_zero_kernel3x3)->RangeMultiplier(bm_config::range_multiplier<ValueType, 2, host_tag>())->Range(bm_config::min_shape<ValueType, 2, host_tag>(), bm_config::max_shape<ValueType, 2, host_tag>())->UseRealTime();

BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(byte)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(int16_t)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(int32_t)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(int64_t)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(float)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(double)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(point4f)
BM_TENSOR_RANK2_CONV_LAZY_ARRAY_INDEX_INSIDE_CLAMP_ZERO_KERNEL3x3(hete_float32x4_t)
