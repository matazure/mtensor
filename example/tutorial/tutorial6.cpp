#include <matazure/tensor>
#include <chrono>

using namespace matazure;

namespace immediate {

	template <typename _Tensor>
	auto add(_Tensor ts0, _Tensor ts1) {
		_Tensor ts_re(ts0.shape());
		for_index(ts0.shape(), [=](pointi<2> idx) {
			ts_re(idx) = ts0(idx) + ts1(idx);
		});

		return ts_re;
	}

	template <typename _Tensor>
	auto sub(_Tensor ts0, _Tensor ts1) {
		_Tensor ts_re(ts0.shape());
		for_index(ts0.shape(), [=](pointi<2> idx) {
			ts_re(idx) = ts0(idx) - ts1(idx);
		});

		return ts_re;
	}

}

namespace lazy {

	template <typename _Tensor0, typename _Tensor1>
	auto add(_Tensor0 ts0, _Tensor1 ts1) {
		return make_lambda(ts0.shape(), [=](int_t i) {
			return ts0[i] + ts1[i];
		});
	}

	template <typename _Tensor0, typename _Tensor1>
	auto sub(_Tensor0 ts0, _Tensor1 ts1) {
		return make_lambda(ts0.shape(), [=](int_t i) {
			return ts0[i] - ts1[i];
		});
	}

}

int main() {
	{
		auto t0 = std::chrono::high_resolution_clock::now();

		tensor<float, 2> ts2f0(pointi<2>{10000, 10000});
		tensor<float, 2> ts2f1(ts2f0.shape());
		tensor<float, 2> ts2f2(ts2f0.shape());
		auto lts2f_re0 = lazy::add(ts2f0, ts2f1);
		auto lts2f_re1 = lazy::sub(lts2f_re0, ts2f2);
		auto lts2f_re2 = lazy::add(lts2f_re0, lts2f_re1);
		auto ts_re = lts2f_re2.persist();

		auto t1 = std::chrono::high_resolution_clock::now();
		printf("lazy evaluation cost time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	}

	{
		auto t0 = std::chrono::high_resolution_clock::now();

		tensor<float, 2> ts2f0(pointi<2>{10000, 10000});
		tensor<float, 2> ts2f1(ts2f0.shape());
		tensor<float, 2> ts2f2(ts2f0.shape());
		auto ts2f_re0 = immediate::add(ts2f0, ts2f1);
		auto ts2f_re1 = immediate::sub(ts2f_re0, ts2f2);
		auto ts2f_re2 = immediate::add(ts2f_re0, ts2f_re1);

		auto t1 = std::chrono::high_resolution_clock::now();
		printf("immediate evaluation cost time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	}

	return 0;
}
