mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"  $@

cmake --build . -- -j|| exit 1
