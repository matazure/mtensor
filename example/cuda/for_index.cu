
#include <matazure/tensor>
using namespace matazure;

int main(){
	//申请设备端tensor
	cuda::tensor<float, 1> ts0(5);
	cuda::tensor<float, 1> ts1(ts0.shape());
	//为tensor赋值
	//__matazure__关键字用于声明此lambda算子可以在cuda中运行
	cuda::for_index(0, ts0.size(), [=] __matazure__ (int_t i){
		ts0[i] = static_cast<float>(i);
		ts1[i] = i * 0.1f;
	});
	//将ts0加ts1的结果存入ts2中
	cuda::tensor<float, 1> ts2(ts0.shape());
	cuda::for_index(0, ts0.size(), [=] __matazure__ (int_t i){
		ts2[i] = ts0[i] + ts1[i];
	});
	//打印结果
	cuda::for_index(0, ts2.size(), [=] __matazure__ (int_t i){
		printf("%d : %f\n", i, ts2[i]);
	});
	//等待设备端的任务执行完毕
	cuda::device_synchronize();
	return 0;
}
