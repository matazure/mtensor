
add_subdirectory(host)

if (WITH_CUDA)
    add_subdirectory(cuda)
endif()
