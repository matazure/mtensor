#include <matazure/tensor>

using namespace matazure;

int main() {
	//point<int, 0> p0;  //不可预知的结果

	//point 支持不同元素类型， 不同维度
	point<int, 1> p1 = { 1 };
	point<int, 2> p2 = { 1, 2 };
	point<int, 3> p3 = { 1, 2, 3 };
	point<unsigned int, 3> p3ui;

	//point支持下标操作，和size函数
	point<float, 5> pf5;
	//int_t 相当于 int类型
	for (int_t i = 0; i < pf5.size(); ++i) {
		pf5[i] = i * 0.1f;
	}
	printf("pf5[3] element is %f\n", pf5[3]);

	//point支持常见算术运算（除 postfix++ 和 赋值运算）
	point<double, 2> point1 = { 1.0, 2.0 };
	point<double, 2> point2 = { 10.0, 20.0 };
	auto point_add = point1 + point2;
	printf("point_add: (%f, %f)\n", point_add[0], point_add[1]);
	++point_add;
	//point_add++; //不支持
	//point_add += point1; //不支持

	//const
	const point<double, 2> const_point = point1;
	auto tmp = const_point[0];
	//const_point[0] = 0.0; //不可变值

	//整个point的功能并不多，可以在matazure/point.hpp查看具体支持的运算和操作
	return 0;
}
