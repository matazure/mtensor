
add_executable(ut_cuda_mtensor 
    main.cu
    ut_algorithm.cu
    ut_lambda_tensor.cu
    ut_execution_policy.cu
    ut_tensor.cu
)
target_link_libraries(ut_cuda_mtensor mtensor gtest)
