cmake_minimum_required(VERSION 3.0)

project(MatazureTensor)

option(WITH_CUDA "Enable CUDA" ON)
# option(WITH_BENCHMARK "Enable Benchmark" OFF) #unneccesary

if (WITH_CUDA)
	if(NOT(CMAKE_SIZEOF_VOID_P EQUAL 8))
		message(FATAL_ERROR "Matazure Tensor with CUDA Only support 64 bit platform")
	endif()
endif()

include_directories(.)

set_property( GLOBAL PROPERTY USE_FOLDERS ON)

if (WITH_CUDA)
	find_package(CUDA 8.0 REQUIRED)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda -Wno-deprecated-gpu-targets")
	if (UNIX)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
	endif()
	if (MSVC)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -x cu -Xcompiler \"/wd4819 /wd4503\" ")
	endif()

	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cu)
	foreach(File ${Files})
		get_filename_component(FileName ${File} NAME_WE)
		CUDA_ADD_EXECUTABLE(${FileName} ${File})
	endforeach()
else()
	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cu)
	foreach(File ${Files})
		get_filename_component(FileName ${File} NAME_WE)
		add_executable(${FileName} ${File})
		set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
		set_target_properties(${FileName} PROPERTIES LINKER_LANGUAGE CXX)
	endforeach()
endif()

add_subdirectory(benchmark)
