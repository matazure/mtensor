
add_executable(ut_cuda 
    ut_algorithm.cu
    ut_lambda_tensor.cu
)
target_link_libraries(ut_cuda mtensor gtest gtest_main)
