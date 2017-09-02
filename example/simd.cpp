#include <matazure/tensor>

using namespace matazure;

int main(){
	tensor<__m128, 1> ts0(100);
	tensor<__m128, 1> ts1(100);

	for_each(ts0, [](__m128 &e) {
		e = _mm_set_ps(1.0f, 1.0f, 2.0f, 2.0f);
	});
	for_each(ts1, [](__m128 &e){
		e = _mm_set_ps(1.0f, 1.0f, 2.0f, 2.0f);
	});

	auto re = (ts0 + ts1).persist();

	for_each(re, [](__m128 e){
		printf("value : %f, %f, %f, %f \n", e.m128_f32[0], e.m128_f32[1], e.m128_f32[2], e.m128_f32[3]);
	});

	return 0;
}
