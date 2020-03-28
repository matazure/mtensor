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

if (NOT CMAKE_BUILD_TYPE) 
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(WITH_OPENCL "With OpenCL" OFF)
if (WITH_OPENCL)
	find_package(OPENCL REQUIRED)
endif()

option(WITH_CUDA "With CUDA" OFF)
if(WITH_CUDA)
	if (MSVC)
		#use yourself nvcc version
		set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin/nvcc.exe")
	endif()

	enable_language(CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
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

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

option(BUILD_EXAMPLE "Build Example" ON)
option(BUILD_TEST "Build Test" ON)
option(BUILD_BENCHMARK "Build Benchmark" ON)

message("current compiler is ${CMAKE_CXX_COMPILER}")

# include_directories(.)
set_property( GLOBAL PROPERTY USE_FOLDERS ON)


if (WITH_SSE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

execute_process(COMMAND git submodule update --init third_party WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Build gtest's sample programs." FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark)

add_library(mtensor INTERFACE)
target_include_directories(mtensor
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    INTERFACE $<INSTALL_INTERFACE:include>
)

if (BUILD_EXAMPLE)
	add_subdirectory(sample)
endif()

if (BUILD_BENCHMARK)
	add_subdirectory(benchmark)
endif()

if (BUILD_TEST)
	add_subdirectory(test)
endif()

install(DIRECTORY include DESTINATION include)
