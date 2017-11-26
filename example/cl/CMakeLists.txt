
file(GLOB Files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(File ${Files})
	set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
	get_filename_component(FileName ${File} NAME_WE)
	set(TargetName cl_${FileName})
	add_executable(${TargetName} ${File})
	target_compile_definitions(${TargetName} PRIVATE MATAZURE_OPENCL)
endforeach()
