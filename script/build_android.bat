git submodule update --init cmake/android-cmake

if NOT DEFINED ANDROID_NDK (
   echo "Did you set ANDROID_NDK variable?"
   exit 1;
)

if not exist %ANDROID_NDK% (
  echo "Cannot find ndk: did you install it under %ANDROID_NDK%?"
  exit 1;
)

if NOT DEFINED ANDROID_ABI (
  set ANDROID_ABI="arm64-v8a"
  echo "Set ANDROID_ABI %ANDROID_ABI%"
)


if not exist build_android mkdir build_android
cd build_android

cmake .. ^
    -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%/build/cmake/android.toolchain.cmake" ^
    -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%/prebuilt/windows-x86_64/bin/make.exe" ^
    -DANDROID_NDK=%ANDROID_NDK% ^
    -DANDROID_ABI=%ANDROID_ABI% ^
    -DANDROID_ARM_NEON=false ^
    -DANDROID_NATIVE_API_LEVEL=22 ^
    -DANDROID_TOOLCHAIN=clang ^
    -DANDROID_STL="c++_static" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -G "Unix Makefiles" ^
    %* ^
    || exit /b

cmake --build .  -- -j || exit /b
