#include "bm_image_split_channel.hpp"

BENCHMARK(bm_image_split_channel)->RangeMultiplier(2)->Range(32, 2048)->UseRealTime();

BENCHMARK_MAIN()