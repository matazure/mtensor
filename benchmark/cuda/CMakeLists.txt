add_executable(bm_cuda_mtensor 
    bm_main.cu
    bm_binary_operator.cu
    bm_for_index.cu
    bm_algorithm.cu
    bm_view.cu
    bm_execution_policy.cu
    bm_mem_copy.cu
    bm_conv.cu
)
target_link_libraries(bm_cuda_mtensor mtensor benchmark)
