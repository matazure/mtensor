#include <matazure/tensor>

using namespace matazure;

int main() {
	static_tensor<int, dim<3,  3>> sts;

	tensor<int, 2> ts(10, 10);
	fill(ts, 1);
	//按内存顺序打印值
	for (int_t i = 0; i < ts.size(); ++i) {
		printf("%d, ", ts[i]);
	}
	printf("\n==================================\n");

	//每个元素置为10
	for_each(ts, [](int &e) {
		e = 10;
	});
	for (int_t i = 0; i < ts.size(); ++i) {
		printf("%d, ", ts[i]);
	}
	printf("\n");

	tensor<int, 2> ts2(ts.extent());
	copy(ts, ts2);
	auto re = reduce(ts2, 0, [](int lhs, int rhs) {
		return lhs + rhs;
	});
	printf("reduce ts2 result: %d\n", re);

#ifdef MATAZURE_CUDA
	cu_tensor<int, 2> cts(10, 10);
	fill(cts, 1);

	//每个元素置为10, 设备端lambda需要加上__device__
	for_each(cts, [] __matazure__ (int &e) {
		e = 10;
	});

	cu_tensor<int, 2> cts2(cts.extent());
	copy(cts, cts2);

	tensor<int, 2> ts_tmp(cts2.extent());
	mem_copy(cts2, ts_tmp);
	for (int_t i = 0; i < ts_tmp.size(); ++i) {
		printf("%d, ", ts_tmp[i]);
	}
	printf("\n");
#endif

	return 0;
}
