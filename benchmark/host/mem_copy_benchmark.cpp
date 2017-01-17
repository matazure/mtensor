#include "benchmark/benchmark.h"
#include <list>
#include <vector>

template <typename Container,
          typename ValueType = typename Container::value_type>
static void BM_Sequential(benchmark::State& state) {
  ValueType v = 42;
  while (state.KeepRunning()) {
    Container c;
    for (int i = state.range(0); --i;) c.push_back(v);
  }
  const size_t items_processed = state.iterations() * state.range(0);
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(v));
}
BENCHMARK_TEMPLATE2(BM_Sequential, std::vector<int>, int)
    ->Range(1 << 0, 1 << 10);
BENCHMARK_TEMPLATE(BM_Sequential, std::list<int>)->Range(1 << 0, 1 << 10);

BENCHMARK_MAIN()