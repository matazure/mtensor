add_executable(ut_mtensor_host
    main.cpp
)
target_link_libraries(ut_mtensor_host gtest mtensor)


if (WITH_CUDA)
    add_subdirectory(cuda)
endif()
