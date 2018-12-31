include_directories(simple0 ${OpenCL_INCLUDE_DIRS})
add_executable(simple0 ${CMAKE_CURRENT_SOURCE_DIR}/simple0.cpp)
target_compile_definitions(simple0 PRIVATE MATAZURE_OPENCL)
target_link_libraries(simple0 ${OpenCL_LIBRARIES})
