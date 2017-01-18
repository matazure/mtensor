
# set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Build gtest's sample programs." FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendors/benchmark)
set_property(TARGET benchmark PROPERTY FOLDER "Vendors")

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendors/benchmark/include)

# file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/host/*.cpp)
# foreach(File ${Files})
# 	get_filename_component(FileName ${File} NAME_WE)
# 	add_executable(${FileName} ${File})
# 	add_dependencies(${FileName} benchmark)
# 	target_link_libraries(${FileName} benchmark)
# 	set_property(TARGET ${FileName} PROPERTY FOLDER "BM")
# endforeach()

if(WITH_CUDA)
	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
	foreach(File ${Files})
		get_filename_component(FileName ${File} NAME_WE)
		CUDA_ADD_EXECUTABLE(cu_${FileName} ${File})
		add_dependencies(cu_${FileName} benchmark)
		target_link_libraries(cu_${FileName} benchmark)
		set_property(TARGET cu_${FileName} PROPERTY FOLDER "BM")
	endforeach()
endif()
