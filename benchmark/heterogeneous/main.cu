#include <benchmark/benchmark.h>
#include <matazure/bm_config.hpp>
#include <matazure/tensor>

int main(int argc, char** argv) {
#ifdef USE_CUDA
	matazure::cuda::set_device(0);
#endif
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
