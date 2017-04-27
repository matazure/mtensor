#include <matazure/tensor>

using namespace matazure;

int main() {
	pointi<2> ext{ 4, 8 };
	//tensor构造函数
	tensor<int, 2> ts2(ext);
	//n维tensor也可以通过n个参数来构造
	tensor<float, 2> tsf2(10, 10);

	//返回tensor的extent
	pointi<2> tsf2_ext = tsf2.shape();
	printf("tsf2 extent is (%d, %d)\n", tsf2_ext[0], tsf2_ext[1]);

	for (int_t j = 0; j < tsf2_ext[1]; ++j) {
		for (int_t i = 0; i < tsf2_ext[0]; ++i) {
			//通过pointi来访问tensor
			tsf2(pointi<2>{i, j}) = i + 0.1f * j;
		}
	}

	printf("linear access\n");
	for (int_t i = 0; i < tsf2.size(); ++i) {
		printf("%d: %f\n", i, tsf2[i]);
	}

	printf("array access\n");
	//按数组访问的方式打印值
	for (int_t j = 0; j < tsf2.shape()[1]; ++j) {
		for (int_t i = 0; i < tsf2.shape()[0]; ++i) {
			//通过pointi来访问tensor
			printf("(%d, %d): %f\n", i, j, tsf2(pointi<2>{i, j}));
		}
	}

//定义cuda版的tensor, 当使用nvcc编译器时, MATAZURE_CUDA会被定义
#ifdef MATAZURE_CUDA

	//构造一个tsf2一样大小的cuda::tensor
	cu_tensor<float, 2> cts(tsf2.shape());
	printf("success construct %dx%d cu_tensor.\n", cts.shape()[0], cts.shape()[1]);
	//cts[100] = 3; //runtime error! cu_tensor只有在device函数里可以访问数据
	//将主机端的tsf2值拷贝到cts
	mem_copy(tsf2, cts);

	cu_tensor<float, 2> cts2(cts.shape());
	//设备到设备的值拷贝
	mem_copy(cts, cts2);

	tensor<float, 2> ts_tmp(cts2.shape());
	//将设备端cts值拷贝到主机端ts_temp
	mem_copy(cts2, ts_tmp);

	//ts_temp的输出和上面的tsf2一样
	for (int_t i = 0; i < ts_tmp.size(); ++i) {
		printf("%d: %f\n", i, ts_tmp[i]);
	}

#endif

	return 0;
}