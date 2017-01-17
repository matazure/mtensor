
//#define BENCHMARK_HAS_CXX11

#include <benchmark/benchmark.h>
#include <matazure/tensor>

using namespace matazure;

class BM_linear_lambda_tensor : public ::benchmark::Fixture {
public:
	void SetUp(const ::benchmark::State& state) {
		/*uq_tsf1.reset(new tensor<float, 1>(state.range(0)));
		fill(*uq_tsf1, 1.0f);*/
	}

	void TearDown(const ::benchmark::State& state) { }

	std::unique_ptr<tensor<float, 1>> uq_tsf1;
};

BENCHMARK_F(BM_linear_lambda_tensor, Mul)(benchmark::State &st) {
	tensor<float, 1> tsf1(st.range(0));
	while (st.KeepRunning()) {
		auto tsf1_re = make_lambda(tsf1.extent(), [tsf1](int_t i) {
			return 2.0f * tsf1[i];
		});
	}
}



BENCHMARK_REGISTER_F(BM_linear_lambda_tensor, Mul)->Range(;


BENCHMARK_MAIN()



