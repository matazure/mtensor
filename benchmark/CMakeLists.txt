include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

message("zzm : ${configuration}")

ExternalProject_Add(google_benchmark
	GIT_REPOSITORY https://github.com/Matazure/benchmark
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DBENCHMARK_ENABLE_TESTING=OFF
	INSTALL_COMMAND
		cmake --build . --target install --config --config Release
	UPDATE_COMMAND ""
)

# set(BENCHMARK_ENABLE_TESTING OFF)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendors/benchmark)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/host/*.cpp)
foreach(File ${Files})
	get_filename_component(FileName ${File} NAME_WE)
	add_executable(${FileName} ${File})
	# add_dependencies(${FileName} benchmark)
	# target_link_libraries(${FileName} benchmark)
	set_property(TARGET ${FileName} PROPERTY FOLDER "Benchmark")
endforeach()

if(WITH_CUDA)
	file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu)
	foreach(File ${Files})
		get_filename_component(FileName ${File} NAME_WE)
		CUDA_ADD_EXECUTABLE(cu_${FileName} ${File})
		# add_dependencies(cu_${FileName} benchmark)
		# target_link_libraries(cu_${FileName} benchmark)
		set_property(TARGET cu_${FileName} PROPERTY FOLDER "Benchmark")
	endforeach()
endif()
