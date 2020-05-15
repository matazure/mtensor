add_executable(ut_host_mtensor
    main.cpp
    ut_point.cpp
    ut_local_tensor.cpp
    view/ut_gather.cpp
    view/ut_slice.cpp
    view/ut_permute.cpp
    view/ut_zip.cpp
)
target_link_libraries(ut_host_mtensor gtest mtensor)


if (WITH_CUDA)
    add_subdirectory(cuda)
endif()
