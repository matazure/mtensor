
add_executable(bm_mtensor_host bm_algorithm.cpp)
target_link_libraries(bm_mtensor_host mtensor benchmark)
