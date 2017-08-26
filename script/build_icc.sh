mkdir -p build_icc
cd build_icc
CXX=icc cmake .. -DCMAKE_BUILD_TYPE=Release

if [ "$(uname)" = 'Darwin' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
    cmake --build . -- "-j$(nproc)"
fi
