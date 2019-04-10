#include <matazure/tensor>

using namespace matazure;

int main(){
	//默认按串行执行， index是按顺序的
	for_index(0, 20, [](int_t i) {
		printf("%d ", i);
	});
	//华丽丽的分割线
	printf("\n-------------------\n");

	tensor<byte, 1> ts_vect(100);

	MATAZURE_AUTO_VECTORISED
	for (int_t i = 0, size = ts_vect.size(); i < size; ++i){
		ts_vect[0] += 1;
	}

	//开启omp并行执行， index是不确定的
#ifdef MATAZURE_OPENMP
	for_index(omp_policy{}, 0, 20, [](int_t i) {
		printf("%d ", i);
	});
#endif

	printf("\n");

	return 0;
}
