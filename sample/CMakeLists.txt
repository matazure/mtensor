add_library(image_helper INTERFACE)
target_include_directories(image_helper INTERFACE  ${PROJECT_SOURCE_DIR}/third_party/stb)

if (WITH_CUDA)
    add_executable(sample_cuda_for_index sample_for_index.cu)
    target_link_libraries(sample_cuda_for_index mtensor)
    target_compile_options(sample_cuda_for_index PRIVATE --expt-extended-lambda)


    add_executable(sample_cuda_mandelbrot sample_mandelbrot.cu)
    target_link_libraries(sample_cuda_mandelbrot mtensor image_helper)

    add_executable(sample_cuda_convolution sample_convolution.cu)
    target_link_libraries(sample_cuda_convolution mtensor image_helper)
    target_compile_options(sample_cuda_convolution PRIVATE --expt-extended-lambda)

    add_executable(sample_cuda_matrix_mul sample_matrix_mul.cu)
    target_link_libraries(sample_cuda_matrix_mul mtensor image_helper)
    target_compile_options(sample_cuda_matrix_mul PRIVATE --expt-extended-lambda)


    add_executable(sample_cuda_level_set sample_level_set.cu)
    target_link_libraries(sample_cuda_level_set mtensor image_helper)
    target_compile_options(sample_cuda_level_set PRIVATE --expt-extended-lambda)

    add_executable(sample_cuda_keynote sample_keynote.cu)
    target_link_libraries(sample_cuda_keynote mtensor image_helper)
    if (WITH_OPENMP)
        target_link_libraries(sample_cuda_keynote mtensor gomp)
    endif()
endif ()



