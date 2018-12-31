#include <matazure/tensor>

using namespace matazure;

int main() {
	auto fun = [](pointi<2> idx)->int{
		printf("invoke ts_lambda by idx (%d, %d)\n", idx[0], idx[1]);
		return idx[0] + idx[1];
	};
	auto shape = pointi<2>{ 4, 4 };
	auto lts = make_lambda(shape, fun);
	auto tmp = lts(pointi<2>{2, 2});

	//persist lts to ts0
	tensor<int, 2> ts0(lts.shape());
	for_index(lts.shape(), [=](pointi<2> idx) {
		ts0(idx) = lts(idx);
	});
	//presist lts to ts1
	auto ts1 = lts.persist();

	return 0;
}
