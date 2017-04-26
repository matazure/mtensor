#include <matazure/tensor>

using namespace matazure;

int main(int argc, char *argv[]) {

	tensor<point<byte, 3>, 2> ts_rgb(512, 512);
	io::read_raw_data("data/lena_rgb888_512x512.raw_data", ts_rgb);
	
	auto t = ts_rgb[pointi<2>{2, 3}];

	return 0;
}
