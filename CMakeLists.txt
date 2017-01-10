cmake_minimum_required(VERSION 3.0)

project(MatazureTensor)

if (MSVC)
	message(${CMAKE_GENERATOR})
	if (NOT (${CMAKE_GENERATOR} STREQUAL  "Visual Studio 14 2015 Win64"))
		message(FATAL_ERROR "Matazure Tensor Only support 64 bit platform")
	endif()
endif()

message(${CMAKE_GENERATOR})

find_package(CUDA 8.0 REQUIRED)
include_directories(.)
# include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
if (UNIX)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
endif()

message(${CUDA_NVCC_FLAGS})

if(NOT(CMAKE_SIZEOF_VOID_P EQUAL 8))
message(FATAL_ERROR "Matazure Tensor Only support 64 bit platform")
endif()

file(GLOB ExampleCuFiles ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cu)
foreach(ExampleCuFile ${ExampleCuFiles})
	get_filename_component(ExampleCuFileName ${ExampleCuFile} NAME_WE)
	#auto add cuda include directory libraries
	CUDA_ADD_EXECUTABLE(${ExampleCuFileName} ${ExampleCuFile})
endforeach()
