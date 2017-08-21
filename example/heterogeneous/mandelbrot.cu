#include <matazure/tensor>

#define STB_IMAGE_WRITE_IMPLEMENTATION /
#include <stb_image_write.h>

using namespace matazure;

#ifdef USE_CUDA
#ifndef MATAZURE_CUDA
#error "does not support cuda"
#endif
#endif

#ifdef _OPENMP
#define USE_OPENMP
#endif

typedef point<byte, 3> rgb;

int main(int argc, char *argv[]) {
	auto output_mandelbrot_path = argc > 1 ? argv[1] : "mandelbrot.png";

	//设置最大迭代次数， 超过max_iteration认为不收敛
	int_t max_iteration = 256 * 16;
	//颜色映射函数
	auto color_fun = [max_iteration] __matazure__ (int_t i) -> rgb {
		//t为i的归一化结果
		float t = float(i) / max_iteration;
		auto r = static_cast<byte>(36*(1-t)*t*t*t*255);
		auto g = static_cast<byte>(60*(1-t)*(1-t)*t*t*255);
		auto b = static_cast<byte>(38*(1-t)*(1-t)*(1-t)*t*255);
		return rgb{ r, g, b };
	};
	//区域大小
	pointi<2> shape = { 2048, 2048 };
	//曼德勃罗算子
	auto mandelbrot_fun = [=] __matazure__ (pointi<2> idx)->rgb {
		pointf<2> c = point_cast<float>(idx) / point_cast<float>(shape) * pointf<2>{3.25f, 2.5f} -pointf<2>{2.0f, 1.25f};
		auto z = pointf<2>::all(0.0f);
		auto norm = 0.0f;
		int_t i = 0;
		while (norm <= 4.0f && i < max_iteration) {
			float tmp = z[0] * z[0] - z[1] * z[1] + c[0];
			z[1] = 2 * z[0] * z[1] + c[1];
			z[0] = tmp;
			++i;
			norm = z[0] * z[0] + z[1] * z[1];
		}
		//直接返回rgb的像素值
		return color_fun(i);
	};

#ifdef USE_CUDA
	//通过shape和mandelbrot_fun构造lambda tensor并直接把结果固化的内存（显存）里
	auto cts_mandelbrot_rgb = make_lambda(shape, mandelbrot_fun, device_tag{}).persist();
	auto ts_mandelbrot_rgb = mem_clone(cts_mandelbrot_rgb, host_tag{});
#else
#ifdef USE_OPENMP
	//使用omp_policy并行策略
	auto ts_mandelbrot_rgb = make_lambda(shape, mandelbrot_fun, host_tag{}).persist(omp_policy{});
#else
	auto ts_mandelbrot_rgb = make_lambda(shape, mandelbrot_fun, host_tag{}).persist();
#endif
#endif

	stbi_write_png(output_mandelbrot_path, shape[0], shape[1], 3, ts_mandelbrot_rgb.data(), shape[0] * 3);

	return 0;
}
