#!/bin/bash

# if [ -z "$ANDROID_NDK" ]; then
#     echo "Did you set ANDROID_NDK variable?"
#     exit 1
# fi
# 
# if [ -d "$ANDROID_NDK" ]; then
#     echo "Using Android ndk at $ANDROID_NDK"
# else
#     echo "Cannot find ndk: did you install it under $ANDROID_NDK?"
#     exit 1
# fi

if [ -z "$ANDROID_ABI" ]; then
    export ANDROID_ABI="arm64-v8a with NEON"
    echo "Set ANDROID_ABI $ANDROID_ABI"
fi

source  /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform android
source  /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform android

mkdir -p build_android_icc
cd build_android_icc

cmake .. \
    -DCMAKE_AR=/opt/intel/compilers_and_libraries_2017.4.196/linux/bin/intel64/xiar \
    -DCMAKE_C_COMPILER=/opt/intel/compilers_and_libraries_2017.4.196/linux/bin/intel64/icc \
    -DCMAKE_CXX_COMPILER=/opt/intel/compilers_and_libraries_2017.4.196/linux/bin/intel64/icc \
    -DCMAKE_TOOLCHAIN_FILE=../vendor/android-cmake/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=$ANDROID_ABI \
    -DANDROID_NATIVE_API_LEVEL=21 \
    -DWITH_CUDA=OFF \
    $@ \
    || exit 1

# Cross-platform parallel build
if [ "$(uname)" = 'Darwin' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)" || exit 1
else
    cmake --build . -- "-j$(nproc)" || exit 1
fi
