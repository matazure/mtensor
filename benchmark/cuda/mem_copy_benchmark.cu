#include <matazure/tensor>
#include <chrono>

using namespace matazure;
using namespace std::chrono;

int main() {
	tensor<float, 1> ts_src(1024 * 1024);
	tensor<float, 1> ts_dst(ts_src.extent());

	auto t0 = high_resolution_clock::now();

	mem_copy(ts_src, ts_dst);

	auto t1 = high_resolution_clock::now();

	auto all_bytes_g = ts_src.size() * sizeof(ts_src[0]) / double(1 << 30);
	auto cost_time_s = (t1 - t0).count() / 1000.0 / 1000.0 / 1000.0;

	printf("%f G per seconds.", all_bytes_g / cost_time_s);
}