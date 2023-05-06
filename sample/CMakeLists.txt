add_library(image_helper INTERFACE)
target_include_directories(image_helper INTERFACE  ${PROJECT_SOURCE_DIR}/third_party/stb)

add_executable(sample_mandelbrot sample_mandelbrot.cpp)
target_link_libraries(sample_mandelbrot mtensor image_helper)

if (WITH_CUDA)
    add_executable(sample_cuda_mandelbrot sample_mandelbrot.cu)
    target_link_libraries(sample_cuda_mandelbrot mtensor image_helper)
endif ()



