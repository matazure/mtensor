
add_library(SerializeUT STATIC ${CMAKE_CURRENT_SOURCE_DIR}/serialize_ut.cpp)
target_link_libraries(SerializeUT PRIVATE gtest)
