#include <matazure/tensor>

using namespace matazure;

int main() {

	printf("print linear-index for_index index\n");
	for_index(5, [](int_t i) {
		printf("%d\n", i);
	});

	printf("print array-index for_index index\n");
	for_index(pointi<2>{2, 2}, [](pointi<2> idx) {
		printf("%d, %d\n", idx[0], idx[1]);
	});

	tensor<int, 2>  ts2f(pointi<2>{10, 10});
	for_index(ts2f.shape(), [ts2f](pointi<2> idx) {
		ts2f(idx) = idx[0] * 10 + idx[1];
	});
	for_index(ts2f.shape(), [ts2f](pointi<2> idx) {
		printf("(%d, %d) : %d\n", idx[0], idx[1], ts2f(idx));
	});

	return 0;

}
