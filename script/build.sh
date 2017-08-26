mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Cross-platform parallel build
if [ "$(uname)" = 'Darwin' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
    cmake --build . -- "-j$(nproc)"
fi
