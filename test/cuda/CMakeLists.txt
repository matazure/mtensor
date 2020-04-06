
add_executable(ut_cuda 
    main.cu
    ut_algorithm.cu
    ut_lambda_tensor.cu
    ut_execution_policy.cu
)
target_link_libraries(ut_cuda mtensor gtest)
