#!/bin/bash

mkdir -p build-linux-arm
cd build-linux-arm

cmake .. \
	-DCMAKE_TOOLCHAIN_FILE=cmake/gcc-linaro-arm-linux.toolchain.cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TEST=ON \
	$@ \
	|| exit 1

cmake --build .   -- -j || exit 1

cd ../
