#!/bin/bash

mkdir -p build-linux-aarch64
cd build-linux-aarch64

cmake .. \
	-DCMAKE_TOOLCHAIN_FILE=cmake/gcc-linaro-aarch64-linux.toolchain.cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TEST=ON \
	$@ \
	|| exit 1

cmake --build .   -- -j || exit 1

cd ../
