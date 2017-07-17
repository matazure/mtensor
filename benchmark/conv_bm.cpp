#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

void bm_cu_conv_global(benchmark::State& state){
	pointi<2> ext;
	fill(ext, state.range(0));
	tensor<float, 2> ts_input(ext);
	tesnor<float, 2> ts_output(ts_src.shape());
	static_tensor<float, dim<3,3>> kenel;

	while (state.KeepRunning()){
		copy(puzzle::conv_general(ts_input, kenel), ts_output);
	}

	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
}
