#include <benchmark/benchmark.h>
#include <matazure/bm_config.hpp>

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
