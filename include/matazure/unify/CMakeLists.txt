# if (NOT WITH_CUDA)
# 	set_source_files_properties(tensor.ut.cu  PROPERTIES LANGUAGE CXX)
# endif()

if (WITH_CUDA)
	add_executable(unify.tensor.ut tensor.ut.cu)
	target_link_libraries(unify.tensor.ut)
endif()
