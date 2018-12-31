#include <matazure/tensor>

using namespace matazure;

struct add_operator {
	int operator()(int a, int b) const {
		return a + b;
	}
};

int add(int a, int b) {
	return a + b;
}

int main() {
	auto add_lambda = [](int a, int b)->int{
		return a + b;
	};

	add_operator add_op;

	assert(add(3, 4) == 7);
	assert(add_op(3, 4) == 7);
	assert(add_lambda(3, 4) == 7);

	return 0;
}
