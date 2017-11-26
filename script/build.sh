mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"  $@

# Cross-platform parallel build
if [ "$(uname)" = 'Darwin' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)" || exit 1
else
    cmake --build . -- "-j$(nproc)" || exit 1
fi
