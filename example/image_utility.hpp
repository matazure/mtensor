#pragma once

#ifndef MATAZURE_EXAMPLE_DISABLE_STB_IMPLEMENTATION
	#define STB_IMAGE_IMPLEMENTATION
	#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#include <stb_image.h>
#include <stb_image_write.h>
#include <matazure/tensor>
#include <stdexcept>

inline matazure::tensor<matazure::pointb<3>, 2> read_rgb_image(const char *image_path){
	using namespace matazure;

	pointi<3> shape{};
	auto data = stbi_load(image_path, &shape[1], &shape[2], &shape[0], false);
	if (shape[0] != 3 || !data) {
		printf("need a 3 channel image");
		throw std::runtime_error("only support 3 channel rgb image");
	}

	typedef point<byte, 3> rgb;
	tensor<rgb, 2> ts_rgb(pointi<2>{shape[1], shape[2]}, shared_ptr<rgb>(reinterpret_cast<rgb *>(data), [ ](rgb *p) {
		stbi_image_free(p);
	}));

	return ts_rgb;
}

void write_rgb_png(const char *image_path, matazure::tensor<matazure::pointb<3>, 2> ts_rgb){
	auto re = stbi_write_png(image_path, ts_rgb.shape()[0], ts_rgb.shape()[1], 3, ts_rgb.data(), ts_rgb.shape()[0] * 3);
	if (re == 0){
		throw std::runtime_error("failed to write ts_rgb to png image");
	}
}
