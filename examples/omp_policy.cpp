#include <matazure/tensor>

using namespace matazure;

int main(){
	//默认按串行执行， index是按顺序的
	for_index(0, 10, [](int_t i) {
		printf("%d ", i);
	});

	printf("\n-------------------\n");

	//开启omp并行执行， index是不确定的
	for_index(omp_policy{}, 0, 10, [](int_t i) {
		printf("%d ", i);
	});

	printf("\n");
	return 0;
}
