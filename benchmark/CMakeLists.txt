# # include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/benchmark/include)
# # include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# 
# file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# set_source_files_properties(${Files} PROPERTIES LANGUAGE CXX)
# add_executable(tensor_benchmark ${Files})
# add_dependencies(tensor_benchmark benchmark)
# target_link_libraries(tensor_benchmark benchmark)
# set_target_properties(tensor_benchmark PROPERTIES LINKER_LANGUAGE CXX)
# set_property(TARGET tensor_benchmark PROPERTY FOLDER "benchmark")
# 
# file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous/*.cu)
# set_source_files_properties(${Files} PROPERTIES LANGUAGE CXX)
# if (WITH_CUDA)
# 	CUDA_ADD_EXECUTABLE(hete_host_tensor_benchmark ${Files} OPTIONS -DUSE_HOST)
# else()
# 	add_executable(hete_host_tensor_benchmark ${Files})
# endif()
# add_dependencies(hete_host_tensor_benchmark benchmark)
# target_link_libraries(hete_host_tensor_benchmark benchmark)
# set_target_properties(hete_host_tensor_benchmark PROPERTIES LINKER_LANGUAGE CXX)
# target_compile_definitions(hete_host_tensor_benchmark PRIVATE USE_HOST)
# set_property(TARGET hete_host_tensor_benchmark PROPERTY FOLDER "benchmark")
# 
# if(WITH_CUDA)
# 	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous/*.cu)
# 	CUDA_ADD_EXECUTABLE(hete_cu_tensor_benchmark ${Files} OPTIONS -DUSE_CUDA)
# 	add_dependencies(hete_cu_tensor_benchmark benchmark)
# 	target_link_libraries(hete_cu_tensor_benchmark benchmark)
# 	set_property(TARGET hete_cu_tensor_benchmark PROPERTY FOLDER "benchmark")
# endif()

add_executable(bm_image_split_channel bm_image_split_channel.cpp)
target_link_libraries(bm_image_split_channel tensor benchmark)

if (CMAKE_SYSTEM_PROCESSOR EQUAL aarch64)

add_executable(bm_conv bm_conv.cpp)
target_link_libraries(bm_conv tensor benchmark)

add_executable(bm_matrix_multiply bm_matrix_multiply.cpp)
target_link_libraries(bm_matrix_multiply tensor benchmark)

add_executable(bm_cpu_fp bm_cpu_fp.cpp)
target_link_libraries(bm_cpu_fp tensor benchmark)

endif()


