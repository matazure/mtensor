execute_process(COMMAND git submodule update --init . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/googletest/googletest)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/googletest/googletest/include)

file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set_source_files_properties(${Files} PROPERTIES LANGUAGE CXX)
add_executable(tensor_test ${Files})
add_dependencies(tensor_test gtest)
target_link_libraries(tensor_test gtest)
set_target_properties(tensor_test PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET tensor_test PROPERTY FOLDER "test")

file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous/*.cu)
set_source_files_properties(${Files} PROPERTIES LANGUAGE CXX)
if(WITH_CUDA)
	CUDA_ADD_EXECUTABLE(hete_host_tensor_test ${Files} OPTIONS -DUSE_HOST)
else()
	add_executable(hete_host_tensor_test ${Files})
endif()
add_dependencies(hete_host_tensor_test gtest)
target_link_libraries(hete_host_tensor_test gtest)
target_compile_definitions(hete_host_tensor_test PRIVATE USE_HOST)
set_target_properties(hete_host_tensor_test PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET hete_host_tensor_test PROPERTY FOLDER "test")

if(WITH_CUDA)
	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous/*.cu)
	CUDA_ADD_EXECUTABLE(hete_cu_tensor_test ${Files} OPTIONS -DUSE_CUDA)
	add_dependencies(hete_cu_tensor_test gtest)
	target_link_libraries(hete_cu_tensor_test gtest)
	set_property(TARGET hete_cu_tensor_test PROPERTY FOLDER "test")

	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu)
	CUDA_ADD_EXECUTABLE(cu_tensor_test ${Files})
	add_dependencies(cu_tensor_test gtest)
	target_link_libraries(cu_tensor_test gtest)
	set_property(TARGET cu_tensor_test PROPERTY FOLDER "test")
endif()


add_subdirectory(external)

add_executable(TensorUT ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(TensorUT SerializeUT)
