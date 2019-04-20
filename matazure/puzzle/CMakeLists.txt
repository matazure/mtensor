if (WITH_CUDA)
	add_subdirectory(cuda)
endif()

add_executable(puzzle.conv.bm conv.bm.cpp)
target_link_libraries(puzzle.conv.bm benchmark)
