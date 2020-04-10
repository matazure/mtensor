add_executable(ut_host_mtensor
    main.cpp
    ut_point.cu
)
target_link_libraries(ut_host_mtensor gtest mtensor)


if (WITH_CUDA)
    add_subdirectory(cuda)
endif()
