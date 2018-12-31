#include <matazure/tensor>

using namespace matazure;

template <typename _Tensor>
auto split(_Tensor ts) {
	pointi<3> plane_shape{ ts.shape()[0], ts.shape()[1], 3 };
	tensor<float, 3> ts_plane(plane_shape);
	for_index(ts.shape(), [=](pointi<2> idx) {
		ts_plane(idx[0], idx[1], 0) = ts(idx)[0];
		ts_plane(idx[0], idx[1], 1) = ts(idx)[1];
		ts_plane(idx[0], idx[1], 2) = ts(idx)[2];
	});

	return ts_plane;
}

int main() {
	tensor<point<byte, 3>, 2> ts_bgr(pointi<2>{256, 256});
	point<byte, 3> mean{ 110, 107, 125 };
	point<float, 3> scale{ 0.125f, 0.125f, 0.125f };
	auto lts_bgr_normalize = cast<point<float, 3>>(ts_bgr - mean) * scale;
	tensor<float, 3> ts_plane = split(lts_bgr_normalize);

	return 0;
}
