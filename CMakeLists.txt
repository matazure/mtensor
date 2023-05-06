cmake_minimum_required(VERSION 3.14)

project(Matazure.Tensor CXX)

if (NOT CXX_STANDARD)
	set(CXX_STANDARD 11)
endif()

set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CUDA_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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


add_library(mtensor INTERFACE)
target_include_directories(mtensor
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    INTERFACE $<INSTALL_INTERFACE:include>
)

add_subdirectory(sample)

install(DIRECTORY include DESTINATION include)
