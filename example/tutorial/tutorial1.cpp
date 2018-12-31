#include <matazure/tensor>

using namespace matazure;

int main() {
	//point<int, 0> //error
	point<int, 1> vec1b;
	point<int, 2> vec2b;
	point<float, 3> vec3f;
	point<point<double, 2>, 4> vec4vec2d;

	static_assert(vec1b.size() == 1, "");
	static_assert(vec2b.size() == 2, "");
	static_assert(vec3f.size() == 3, "");

	//arithmetic
	point<float, 2> p0{ 0.0f, 0.0f };
	point<float, 2> p1{ 100.0f, 100.0f };

	auto center = (p0 + p1) / 2.0f;
	auto diff = (p1 - p0);

	point<int, 2> p2{ 100, 100 };
	//auto center = (p0 + p2) / 2.0f; compile error, strong typing

	return 0;
}
