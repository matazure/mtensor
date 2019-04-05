#include <matazure/unify/tensor.hpp>
#include <variant>

using namespace matazure;

int main(int argc, char *argv[]) {

	tensor<float, 2> ts(pointi<2>{10, 10});

	unify::tensor<float, 2> uts(ts);

	//stride(uts);

	return 0;
}
