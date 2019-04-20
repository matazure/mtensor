cmake_minimum_required(VERSION 3.8)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

project(Matazure.Tensor CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WITH_OPENCL "With OpenCL" OFF)
if (WITH_OPENCL)
	find_package(OPENCL REQUIRED)
endif()

option(WITH_CUDA "With CUDA" OFF)
if(WITH_CUDA)
	enable_language(CUDA)
	find_package(CUDA REQUIRED)
endif()

option(WITH_OPENMP "With OpenMP" OFF)
if(WITH_OPENMP)
	find_package(OpenMP REQUIRED)
	# clang not surpport
	# if(NOT OPENMP_FOUND)
	# 	message(FATAL_ERROR "OPENMP_FOUND is not found")
	# endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

	if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	endif()
endif()

option(BUILD_EXAMPLE "Build Example" ON)
option(BUILD_TEST "Build Test" ON)
option(BUILD_BENCHMARK "Build Benchmark" ON)

message("current compiler is ${CMAKE_CXX_COMPILER}")

include_directories(.)
set_property( GLOBAL PROPERTY USE_FOLDERS ON)

set(BUILD_SHARED_LIBS OFF)
if (MSVC)
	add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj")

	foreach (flag_var
					 CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
					 CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
		if (NOT BUILD_SHARED_LIBS AND NOT gtest_force_shared_crt)
			string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
		endif()
	endforeach()
endif()

if (UNIX)
	add_definitions("-x c++")
endif()

if (WITH_SSE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

if (WITH_CUDA)
	find_package(CUDA 9.0 REQUIRED)

	if(NOT(CMAKE_SIZEOF_VOID_P EQUAL 8))
		message(FATAL_ERROR "Matazure Tensor with CUDA Only support 64 bit platform")
	endif()

	#--expt-relaxed-constexpr  (don't use this)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda --disable-warnings")
	if (UNIX)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++14")
	endif()
	if (MSVC)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd4819 /wd4503\" ")
	endif()

	#disable sse,because the cuda 9.0 bug with sse
	#add_definitions(-DMATAZURE_DISABLE_SSE)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

execute_process(COMMAND git submodule update --init third_party WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Build gtest's sample programs." FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark)

add_subdirectory(matazure)

add_subdirectory(dependency)

if (BUILD_EXAMPLE)
	add_subdirectory(example)
endif()

if (BUILD_BENCHMARK)
	add_subdirectory(benchmark)
endif()

if (BUILD_TEST)
	add_subdirectory(test)
endif()
