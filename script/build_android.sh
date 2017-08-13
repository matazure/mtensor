#!/bin/bash

if [ -z "$ANDROID_NDK" ]; then
    echo "Did you set ANDROID_NDK variable?"
    exit 1
fi

if [ -d "$ANDROID_NDK" ]; then
    echo "Using Android ndk at $ANDROID_NDK"
else
    echo "Cannot find ndk: did you install it under $ANDROID_NDK?"
    exit 1
fi

mkdir -p build_android
cd build_android

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../vendor/android-cmake/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="x86_64" \
    -DANDROID_NATIVE_API_LEVEL=21 \
    -DWITH_CUDA=OFF \
    $@ \
    || exit 1

# Cross-platform parallel build
if [ "$(uname)" = 'Darwin' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
    cmake --build . -- "-j$(nproc)"
fi
