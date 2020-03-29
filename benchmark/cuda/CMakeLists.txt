add_executable(bm_cuda bm_algorithm.cu)
target_link_libraries(bm_cuda mtensor benchmark)
