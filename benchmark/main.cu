#include <benchmark/benchmark.h>
#include <matazure/tensor>

int main(int argc, char** argv) {
	matazure::cuda::set_device(1);
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
