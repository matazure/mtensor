add_executable(ut_host_mtensor
    main.cpp
    ut_point.cpp
    ut_local_tensor.cpp
    ut_zero_and_one.cpp
    view/ut_gather.cpp
    view/ut_slice.cpp
    view/ut_permute.cpp
    view/ut_zip.cpp
    view/ut_one.cpp
    view/ut_zero.cpp
    view/ut_meshgrid.cpp
    view/ut_eye.cpp
    view/ut_conv.cpp
    view/ut_broadcast.cpp
)
target_link_libraries(ut_host_mtensor gtest mtensor)


if (WITH_CUDA)
    add_subdirectory(cuda)
endif()
