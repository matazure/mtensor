cmake_minimum_required(VERSION 3.8)
#cuda need 3.8
#add_link_options need 3.13

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

if (NOT CXX_STANDARD)
	set(CXX_STANDARD 11)
endif()

set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CUDA_STANDARD ${CXX_STANDARD})
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
	# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -arch=sm_75")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

	if (CUDA_ARCH) 
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${CUDA_ARCH}")
	endif()

endif()

option(WITH_OPENMP "With OpenMP" OFF)
if(WITH_OPENMP)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"-fopenmp\"")
	# need manully set link options
	# add_link_options($<$<COMPILE_LANGUAGE:CUDA>:-lgomp>)
	set(MTENSOR_CUDA_LINK_FLAGS "${LINK_FLAGS} -lgomp")
endif()

option(NATIVE "-march=native" OFF)
if (NATIVE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"-march=native\"")
endif()

option(BUILD_EXAMPLE "Build Example" ON)
option(BUILD_TEST "Build Test" ON)
option(BUILD_BENCHMARK "Build Benchmark" ON)

message("current compiler is ${CMAKE_CXX_COMPILER}")

set_property( GLOBAL PROPERTY USE_FOLDERS ON)


if (WITH_SSE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DOWNLOAD_SUBMODULES "git submodule update --init ." ON)

if (DOWNLOAD_SUBMODULES)
	execute_process(COMMAND git submodule update --init . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

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
