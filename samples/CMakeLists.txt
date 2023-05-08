add_library(image_helper INTERFACE)
target_include_directories(image_helper INTERFACE  ${PROJECT_SOURCE_DIR}/third_party/stb)

add_executable(add add.cpp)
target_link_libraries(add mtensor image_helper)

if (WITH_CUDA)
    add_executable(mandelbrot mandelbrot.cu)
    target_link_libraries(mandelbrot mtensor image_helper)
endif ()



