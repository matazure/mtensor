//#include <benchmark/benchmark.h>
// #include <bm_config.hpp>
//#include <matazure/tensor>
//#include <emmintrin.h>
//
//using namespace matazure;
//
//void bm_conv_direct(benchmark::State &state){
//	pointi<2> ext;
//	fill(ext, state.range(0));
//	tensor<float, 2> ts_input(ext);
//	tensor<float, 2> ts_output(ts_input.shape());
//	static_tensor<float, dim<3,3>> kenel;
//
//	while (state.KeepRunning()){
//		puzzle::conv_direct(ts_input, kenel, ts_output);
//	}
//
//	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
//	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
//}
//BENCHMARK(bm_conv_direct)->RangeMultiplier(2)->Range(14, 224)->UseRealTime();
//
//void bm_conv_general(benchmark::State& state){
//	pointi<2> ext;
//	fill(ext, state.range(0));
//	tensor<float, 2> ts_input(ext);
//	tensor<float, 2> ts_output(ts_input.shape());
//	static_tensor<float, dim<3,3>> kenel;
//
//	while (state.KeepRunning()){
//		copy(puzzle::conv_general(ts_input, kenel), ts_output);
//	}
//
//	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
//	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
//}
//BENCHMARK(bm_conv_general)->RangeMultiplier(2)->Range(14, 224)->UseRealTime();
//
//
//void bm_conv_sse2(benchmark::State &state){
//	pointi<2> ext;
//	fill(ext, state.range(0));
//	tensor<float, 2> ts_input(ext);
//	tensor<float, 2> ts_output(ts_input.shape());
//	static_tensor<float, dim<3,3>> kenel;
//
//	while (state.KeepRunning()){
//		auto kenel_radius = kenel.shape() / 2;
//		//#pragma omp parallel for collapse(2)
//		for (int_t j = 0; j < ts_input.shape()[1]; ++j) {
//			for (int_t i = 4; i < ts_input.shape()[0] - 4; i += 4) {
//				__m128 sum = _mm_setzero_ps();
//
//				__m128 pixel00 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 0} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{0, 0} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{0, 0} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{0, 0} -kenel_radius]);
//				__m128 pixel10 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 0} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{1, 0} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{1, 0} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{1, 0} -kenel_radius]);
//				__m128 pixel20 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 0} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{2, 0} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{2, 0} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{2, 0} -kenel_radius]);
//				__m128 pixel01 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 1} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{0, 1} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{0, 1} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{0, 1} -kenel_radius]);
//				__m128 pixel11 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 1} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{1, 1} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{1, 1} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{1, 1} -kenel_radius]);
//				__m128 pixel21 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 1} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{2, 1} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{2, 1} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{2, 1} -kenel_radius]);
//				__m128 pixel02 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{0, 2} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{0, 2} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{0, 2} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{0, 2} -kenel_radius]);
//				__m128 pixel12 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{1, 2} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{1, 2} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{1, 2} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{1, 2} -kenel_radius]);
//				__m128 pixel22 =  _mm_set_ps(ts_input[pointi<2>{i, j} +pointi<2>{2, 2} -kenel_radius], ts_input[pointi<2>{i+1, j} +pointi<2>{2, 2} -kenel_radius], ts_input[pointi<2>{i+2, j} +pointi<2>{2, 2} -kenel_radius], ts_input[pointi<2>{i+3, j} +pointi<2>{2, 2} -kenel_radius]);
//
//				auto weight00 = _mm_set_ps1(kenel[pointi<2>{0, 0}]);
//				auto weight10 = _mm_set_ps1(kenel[pointi<2>{1, 0}]);
//				auto weight20 = _mm_set_ps1(kenel[pointi<2>{2, 0}]);
//				auto weight01 = _mm_set_ps1(kenel[pointi<2>{0, 1}]);
//				auto weight11 = _mm_set_ps1(kenel[pointi<2>{1, 1}]);
//				auto weight21 = _mm_set_ps1(kenel[pointi<2>{2, 1}]);
//				auto weight02 = _mm_set_ps1(kenel[pointi<2>{0, 2}]);
//				auto weight12 = _mm_set_ps1(kenel[pointi<2>{1, 2}]);
//				auto weight22 = _mm_set_ps1(kenel[pointi<2>{2, 2}]);
//
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel00, weight00));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel10, weight10));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel20, weight20));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel01, weight01));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel11, weight11));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel21, weight21));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel02, weight02));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel12, weight12));
//				sum = _mm_add_ps(sum, _mm_mul_ps(pixel22, weight22));
//
//				auto dst_p = ts_output.data() + index2offset(pointi<2>{i, j}, ts_output.stride(), first_major{});
//				_mm_store_ps(dst_p, sum);
//			}
//		}
//	}
//
//	state.SetBytesProcessed(state.iterations() * ts_output.size() * sizeof(float));
//	state.SetItemsProcessed(state.iterations() * ts_output.size() * kenel.size());
//}
//BENCHMARK(bm_conv_sse2)->RangeMultiplier(2)->Range(16, 224)->UseRealTime();
