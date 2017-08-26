#pragma once

struct bm_config{
	constexpr static int max_host_memory_exponent(){
		return 30;
	}

	constexpr static int max_cuda_memory_exponent(){
		return 30;
	}

	constexpr static int max_cl_memory_exponent(){
		return 30;
	}
};
