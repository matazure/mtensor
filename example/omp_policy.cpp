#include <matazure/tensor>

#ifdef MATAZRE_OPENMP

using namespace matazure;

int main(){
	tensor<float, 2> ts(100, 100);
	//默认按串行执行， index是按顺序的
	for_index(0, 10, [](int_t i) {
		printf("%d ", i);
	});
	//华丽丽的分割线
	printf("\n-------------------\n");
	//开启omp并行执行， index是不确定的
	for_index(omp_policy{}, 0, 10, [](int_t i) {
		printf("%d ", i);
	});
	//防止打出乱码
	printf("\n");
	return 0;
}

#endif
