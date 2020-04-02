add_executable(bm_mtensor_host 
    bm_binary_operator.cpp
    bm_for_index.cpp
    bm_algorithm.cpp
    bm_view.cpp
)
target_link_libraries(bm_mtensor_host mtensor benchmark)

if (WITH_CUDA)
    add_subdirectory(cuda)
endif()
