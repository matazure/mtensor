
add_executable(ut_mtensor_cuda 
    main.cu
    ut_algorithm.cu
    ut_lambda_tensor.cu
    ut_execution_policy.cu
)
target_link_libraries(ut_mtensor_cuda mtensor gtest)
