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

if [ -z "$ANDROID_ABI" ]; then
    # export ANDROID_ABI="armeabi-v7a"
    export ANDROID_ABI="arm64-v8a"
    echo "Set ANDROID_ABI $ANDROID_ABI"
fi

mkdir -p build_android
cd build_android

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_NDK=$ANDROID_NDK \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_TOOLCHAIN=clang \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_NATIVE_API_LEVEL=21 \
    -DBUILD_TEST=ON \
    -DWITH_CUDA=OFF \
    $@ \
    || exit 1


cmake --build . -- -j || exit 1
