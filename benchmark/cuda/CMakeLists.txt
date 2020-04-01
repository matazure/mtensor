add_executable(bm_mtensor_cuda 
    bm_for_index.cu
    bm_algorithm.cu
)
target_link_libraries(bm_mtensor_cuda mtensor benchmark)
