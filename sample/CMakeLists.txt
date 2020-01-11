# add_subdirectory(host)
# add_subdirectory(cuda)

add_executable(sample1 sample1.cpp)
target_link_libraries(sample1 tensor)

add_executable(sample2 sample2.cpp)
target_link_libraries(sample2 tensor)

if (WITH_CUDA)
    add_executable(sample3 sample3.cu)
    target_link_libraries(sample3 tensor)

    add_executable(sample4 sample4.cu)
    target_include_directories(sample4 PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/stb)
    target_link_libraries(sample4 tensor)
endif ()

add_executable(sample5_gradient sample5_gradient.cpp)
target_include_directories(sample5_gradient PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/stb)
target_link_libraries(sample5_gradient tensor)
