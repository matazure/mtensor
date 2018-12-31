#include <matazure/tensor>

using namespace matazure;

int main() {
	//constrcut
	//tensor<int, 1> vec(0);
	tensor<int, 1> vec(pointi<1>{10});
	//tensor<int, 2> mat(10, 10);
	tensor<int, 2> mat(pointi<2>{10, 10});
	//tensor<int, 3> ts(10, 10, 10);
	tensor<int, 3> ts(pointi<3>{10, 10, 10});

	//assign vec by linear-index
	vec(0) = 0;
	//assign vec by array-index
	vec(pointi<1>{0}) = 0;
	// assign mat by linear-index
	mat(99) = 0;
	// assign mat by array-index
	mat(pointi<2>{9, 9}) = 0;
	// assign mat by pretty array-index
	mat(9, 9) = 0;
	//assign ts by linear-index
	ts(9) = 0;
	//assign ts by array-index
	ts(pointi<3>{9, 9, 9}) = 0;
	//assign ts by pretty  array-index
	ts(9, 9, 9) = 0;

	assert(vec.size() == 10);
	assert(vec.shape().size() == 1);
	assert(vec.shape()[0] == 10);

	assert(mat.size() == 100);
	assert(mat.shape().size() == 2);
	assert(mat.shape()[0] == 10);
	assert(mat.shape()[1] == 10);

	assert(ts.size() == 1000);
	assert(ts.shape().size() == 3);
	assert(ts.shape()[0] == 10);
	assert(ts.shape()[1] == 10);
	assert(ts.shape()[2] == 10);

	return 0;
}
