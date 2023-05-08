cmake_minimum_required(VERSION 3.14)

project(MTensor CXX)

set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	set(WITH_CUDA ON)
	set(CMAKE_CUDA_ARCHITECTURES OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
endif()

add_library(mtensor INTERFACE)
target_include_directories(mtensor
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    INTERFACE $<INSTALL_INTERFACE:include>
)

add_subdirectory(samples)
