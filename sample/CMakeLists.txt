# add_subdirectory(host)
# add_subdirectory(cuda)

add_executable(sample1 sample1.cpp)
target_link_libraries(sample1 tensor)

add_executable(sample2 sample2.cpp)
target_link_libraries(sample2 tensor)

add_library(image_helper INTERFACE)
target_include_directories(image_helper INTERFACE  ${PROJECT_SOURCE_DIR}/third_party/stb)

if (WITH_CUDA)
    add_executable(sample3 sample3.cu)
    target_link_libraries(sample3 tensor)

    add_executable(sample4 sample4.cu)
    target_link_libraries(sample4 tensor image_helper)

    add_executable(sample6_convolution sample6_convolution.cu)
    target_link_libraries(sample6_convolution tensor image_helper)

    add_executable(sample7_matrix_mul sample7_matrix_mul.cu)
    target_link_libraries(sample7_matrix_mul tensor image_helper)
endif ()

add_executable(sample5_gradient sample5_gradient.cpp)
target_link_libraries(sample5_gradient tensor image_helper)

