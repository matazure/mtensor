add_executable(bm_host_mtensor 
    bm_main.cpp
    bm_binary_operator.cpp
    bm_for_index.cpp
    bm_algorithm.cpp
    bm_view.cpp
    bm_mem_copy.cpp
)
target_link_libraries(bm_host_mtensor mtensor benchmark)

if (WITH_CUDA)
    add_subdirectory(cuda)
endif()
