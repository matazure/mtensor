#include <benchmark/benchmark.h>
#include <bm_config.hpp>

int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
