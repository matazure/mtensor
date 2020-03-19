mkdir -p build
cd build
cmake ..  $@

cmake --build . -- -j|| exit 1
