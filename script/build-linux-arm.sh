#!/bin/bash

if [ -z "$GCC_LINARO_TOOLCHAIN" ]; then
	# export GCC_LINARO_TOOLCHAIN="gcc-linaro-7.4.1-2019.02-x86_64_arm-linux-gnueabihf"
	echo "Did you set GCC_LINARO_TOOLCHAIN variable?"
	exit 1
fi

if [ -d "$GCC_LINARO_TOOLCHAIN" ]; then
	echo "Using gcc linaro toolchain at $GCC_LINARO_TOOLCHAIN"
else
	echo "Cannot find gcc linaro toolchain: did you install it under $GCC_LINARO_TOOLCHAIN?"
	exit 1
fi


mkdir -p build/linux/arm
cd build/linux/arm

cmake ../../../ \
	-DCMAKE_TOOLCHAIN_FILE=cmake/gcc-linaro-arm-linux.toolchain.cmake \
	-DGCC_LINARO_TOOLCHAIN=$GCC_LINARO_TOOLCHAIN \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TEST=ON \
	$@ \
	|| exit 1

cmake --build .   -- -j || exit 1

cd ../
