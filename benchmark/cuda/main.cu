#include <benchmark/benchmark.h>
#include <bm_config.hpp>
#include <matazure/tensor>

int main(int argc, char** argv) {
	//auto id = matazure::cuda::get_device();
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
