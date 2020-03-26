#!/bin/bash

mkdir -p build-linux-armv7
cd build-linux-armv7

cmake .. \
	-DCMAKE_TOOLCHAIN_FILE=cmake/gcc-linaro-arm-linux.toolchain.cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TEST=ON \
	$@ \
	|| exit 1

cmake --build .   -- -j || exit 1

cd ../
