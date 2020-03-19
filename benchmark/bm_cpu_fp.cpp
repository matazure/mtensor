#include "bm_config.hpp"



void bm_cpu_fq(benchmark::State & state) {
	auto count = state.range(0);

	while (state.KeepRunning()) {
        for (int i = 0; i < count; ++i) {
#ifdef __aarch64__
			__asm(
				"fmla v0.4s, v0.4s, v0.4s\n"
				"fmla v1.4s, v1.4s, v1.4s\n"
				"fmla v2.4s, v2.4s, v2.4s\n"
				"fmla v3.4s, v3.4s, v3.4s\n"
				"fmla v4.4s, v4.4s, v4.4s\n"
				// "fmla v5.4s, v5.4s, v5.4s\n"
				// "fmla v6.4s, v6.4s, v6.4s\n"
				// "fmla v7.4s, v7.4s, v7.4s\n"
				// "fmla v8.4s, v8.4s, v8.4s\n"
				// "fmla v9.4s, v9.4s, v9.4s\n"
			);
#else
		    __asm(
				"vfmadd231ps %ymm0, %ymm0, %ymm0\n"
				"vfmadd231ps %ymm1, %ymm1, %ymm1\n"
				"vfmadd231ps %ymm2, %ymm2, %ymm2\n"
				"vfmadd231ps %ymm3, %ymm3, %ymm3\n"
				"vfmadd231ps %ymm4, %ymm4, %ymm4\n"
				"vfmadd231ps %ymm5, %ymm5, %ymm5\n"
				"vfmadd231ps %ymm6, %ymm6, %ymm6\n"
				"vfmadd231ps %ymm7, %ymm7, %ymm7\n"
				"vfmadd231ps %ymm8, %ymm8, %ymm8\n"
				"vfmadd231ps %ymm9, %ymm9, %ymm9\n"
				"vfmadd231ps %ymm10, %ymm10, %ymm10\n"
				// "vfmadd231ps %ymm11, %ymm11, %ymm11\n"
				// "vfmadd231ps %ymm12, %ymm12, %ymm12\n"
				// "vfmadd231ps %ymm13, %ymm13, %ymm13\n"
				// "vfmadd231ps %ymm14, %ymm14, %ymm14\n"
				// "vfmadd231ps %ymm15, %ymm15, %ymm15\n"
				// "vfmadd231ps %ymm16, %ymm16, %ymm16\n"
				// "vfmadd231ps %ymm17, %ymm17, %ymm17\n"
            );
#endif
		}
	}

	// auto byte_size = (col * row) * sizeof(value_type);
	// auto item_size = col * row * 2 * element_size;
	// state.SetBytesProcessed(state.iterations() * byte_size);
	state.SetItemsProcessed(state.iterations() * count);
}

BENCHMARK(bm_cpu_fq)->Arg(10000);

BENCHMARK_MAIN();