add_executable(puzzle.cuda.conv.bm conv.bm.cu)
target_link_libraries(puzzle.cuda.conv.bm benchmark)

add_executable(puzzle.cuda.prod.bm prod.bm.cu)
target_link_libraries(puzzle.cuda.prod.bm benchmark)
