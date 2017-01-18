
//#define BENCHMARK_HAS_CXX11

#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

class BM_linear_lambda_persist_tensor : public ::benchmark::Fixture {
public:
	void SetUp(const ::benchmark::State& state) {
		uq_tsf1.reset(new tensor<float, 1>(state.range(0)));
		fill(*uq_tsf1, 1.0f);
	}

	void TearDown(const ::benchmark::State& state) { }

	std::unique_ptr<tensor<float, 1>> uq_tsf1;
};

BENCHMARK_F(BM_linear_lambda_persist_tensor, Mul)(benchmark::State &st) {
	while (st.KeepRunning()) {
		auto tfs1 = *uq_tsf1;
		auto tsf1_re = make_lambda(tfs1.extent(), [tfs1](int_t i) {
			return 2.0f * tfs1[i];
		});
	}
}

//BENCHMARK_DEFINE_F(BM_linear_lambda_persist_tensor, Bar)(benchmark::State& st) {
//	if (st.thread_index == 0) {
//		assert(data.get() != nullptr);
//		assert(*data == 42);
//	}
//	while (st.KeepRunning()) {
//		assert(data.get() != nullptr);
//		assert(*data == 42);
//	}
//	st.SetItemsProcessed(st.range(0));
//}

BENCHMARK_REGISTER_F(BM_linear_lambda_persist_tensor, Mul)->Arg(1<<20);


BENCHMARK_MAIN()



