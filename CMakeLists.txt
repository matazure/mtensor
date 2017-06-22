cmake_minimum_required(VERSION 3.1)

project(MatazureTensor CXX)

option(WITH_CUDA "With CUDA" ON)
option(WITH_BENCHMARK "With Benchmark" ON)
option(WITH_OPENMP "With OpenMP" ON)

message("current compiler is ${CMAKE_CXX_COMPILER}")

include_directories(.)
set_property( GLOBAL PROPERTY USE_FOLDERS ON)

if (WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

if (WITH_OPENMP)
	find_package(OpenMP)
	set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

if (UNIX)
	add_definitions("-x c++")
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") #conflict with  CUDA_NVCC_FLAGS, so use target_compile_features
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}-s -O3 -Wall")
endif()

if (WITH_CUDA)
	if(NOT(CMAKE_SIZEOF_VOID_P EQUAL 8))
		message(FATAL_ERROR "Matazure Tensor with CUDA Only support 64 bit platform")
	endif()

	find_package(CUDA 8.0 REQUIRED)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda  -Wno-deprecated-gpu-targets")
	if (UNIX)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
	endif()
	if (MSVC)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd4819 /wd4503\" ")
	endif()
endif()

#add host examples
file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
foreach(File ${Files})
	set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
	get_filename_component(FileName ${File} NAME_WE)
	add_executable(${FileName} ${File})
	set_target_properties(${FileName} PROPERTIES LINKER_LANGUAGE CXX)
	target_compile_features(${FileName} PRIVATE cxx_constexpr cxx_deleted_functions cxx_lambdas cxx_rvalue_references cxx_variadic_templates)
endforeach()

#add heterogeneous examples
if (WITH_CUDA)
	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/examples/heterogeneous/*.cu)
	foreach(File ${Files})
		get_filename_component(FileName ${File} NAME_WE)
		CUDA_ADD_EXECUTABLE(hete_${FileName} ${File})
	endforeach()
else()
	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/examples/heterogeneous/*.cpp)
	foreach(File ${Files})
		set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
		get_filename_component(FileName ${File} NAME_WE)
		set(TargetName hete_${FileName})
		add_executable(${TargetName} ${File})
		set_target_properties(${TargetName} PROPERTIES LINKER_LANGUAGE CXX)
		target_compile_features(${TargetName} PRIVATE cxx_constexpr cxx_deleted_functions cxx_lambdas cxx_rvalue_references cxx_variadic_templates)
	endforeach()
endif()

#add cuda examples
if (WITH_CUDA)
	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/examples/cuda/*.cu)
	foreach(File ${Files})
		get_filename_component(FileName ${File} NAME_WE)
		CUDA_ADD_EXECUTABLE(cu_${FileName} ${File})
	endforeach()
endif()

if (WITH_BENCHMARK)
	add_subdirectory(benchmark)
endif()
