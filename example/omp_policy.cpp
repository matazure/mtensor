#include <matazure/tensor>

using namespace matazure;

int main(){
	//默认按串行执行， index是按顺序的
	for_index(0, 20, [](int_t i) {
		printf("%d ", i);
	});
	//华丽丽的分割线
	printf("\n-------------------\n");

	//开启omp并行执行， index是不确定的
#ifdef MATAZURE_OPENMP
	for_index(omp_policy{}, 0, 20, [](int_t i) {
		printf("%d ", i);
	});
#endif

	printf("\n");

	return 0;
}
