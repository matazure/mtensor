#include <matazure/tensor>

using namespace matazure;

int main(int argc, char *argv[]) {
	//构造一个10x5的二维数组
	constexpr int rank = 2;
	int col = 100;
	int row = 50;
	pointi<rank> shape{col, row};
	tensor<float, rank> ts(shape);

	auto t = clamp_zero(ts);

	// // class halo_functor {
	// // 	[=]
	// // };

	// auto lts = make_lambda(shape, [=](pointi<2> index) {
	// 	if (outside_range(shape, pointi<2>{0, 0}, shape)) {
	// 		return ts(index) - ts(index - 1 );
	// 	} else {
	// 		return t(index) - t(index);
	// 	}
	// });

	// // halo_template(ts, [](pointi<2> index) {

	// // });

	// return 0;
}
