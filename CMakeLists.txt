cmake_minimum_required(VERSION 3.1)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

project(MatazureTensor CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA 9.0)
if(CUDA_FOUND)
	option(WITH_CUDA "With CUDA" ON)
else()
	option(WITH_CUDA "With CUDA" OFF)
endif()
if(WITH_CUDA)
	if(NOT CUDA_FOUND)
		message(FATAL_ERROR "CUDA is not found")
	endif()
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
	option(WITH_OPENMP "With OpenMP" ON)
else()
	option(WITH_OPENMP "With OpenMP" OFF)
endif()
if(WITH_OPENMP)
	# clang not surpport 
	# if(NOT OPENMP_FOUND)
	# 	message(FATAL_ERROR "OPENMP_FOUND is not found")
	# endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")	
	
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")	
	endif()
endif()

option(BUILD_EXAMPLE "Build Example" ON)
option(BUILD_BENCHMARK "Build Benchmark" OFF)
option(BUILD_TEST "Build Test" OFF)

message("current compiler is ${CMAKE_CXX_COMPILER}")

include_directories(.)
set_property( GLOBAL PROPERTY USE_FOLDERS ON)

set(BUILD_SHARED_LIBS OFF)
if (MSVC)
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
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

	if (ANDROID)
		# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=atom -mtune=atom")
	endif()
endif()

if (WITH_CUDA)
	find_package(CUDA 8.0 REQUIRED)

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
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (BUILD_EXAMPLE)
	add_subdirectory(example)
endif()

if (BUILD_BENCHMARK)
	add_subdirectory(benchmark)
endif()

if (BUILD_TEST)
	add_subdirectory(test)
endif()
