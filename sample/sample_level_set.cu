#include <chrono>
#include "image_helper.hpp"
#include "sample_level_set.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample_level_set input_image" << std::endl;
        return -1;
    }

    tensor<byte, 2> img_gray = read_gray_image(argv[1]);
    auto cu_img_gray = identify(img_gray, device_t{});
    auto img_float = view::cast<float>(cu_img_gray).persist();
    auto img_grad = gradient(img_float);
    auto mat_g = view::map(img_grad,
                           [] MATAZURE_GENERAL(point<float, 2> grad) -> float {
                               return 1.0f / (1.0f + grad[0] * grad[0] + grad[1] * grad[1]);
                           })
                     .persist();

    cuda::tensor<float, 2> mat_phi0(img_float.shape());
    float c0 = 2.0f;
    fill(mat_phi0, c0);
    fill(view::slice(mat_phi0, point2i{20, 25}, point2i{10, 10}), -c0);
    fill(view::slice(mat_phi0, point2i{40, 25}, point2i{10, 10}), -c0);

    float lambda = 5;
    float alfa = -3;
    float epsilon = 1.5f;
    float timestep = 1;
    float mu = 0.2 / timestep;

    auto t0 = std::chrono::high_resolution_clock::now();

    auto mat_phi = drlse_edge(mat_phi0, mat_g, lambda, mu, alfa, epsilon, timestep, 100);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "cost time " << (t1 - t0).count() << " ns" << std::endl;

    cuda::tensor<byte, 2> img_phi(mat_phi.shape());
    transform(mat_phi, img_phi, [] MATAZURE_GENERAL(float v) {
        return static_cast<uint8_t>(min(255.0f, max(0.0f, -v * 100.f)));
    });

    write_gray_png("phi.png", identify(img_phi, host_t{}));

    return 0;
}
