add_library(tensor INTERFACE)
target_include_directories(tensor
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    INTERFACE $<INSTALL_INTERFACE:include>
)

# add_subdirectory(unify)
add_subdirectory(simd)

# add_subdirectory(puzzle)
