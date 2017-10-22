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
  set ANDROID_ABI="armeabi-v7a with NEON"
  echo "Set ANDROID_ABI %ANDROID_ABI%"
)

if NOT DEFINED CMAKE_BUILD_TYPE (
  set CMAKE_BUILD_TYPE=Release
)

if not exist build_android mkdir build_android
cd build_android

cmake .. ^
    -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%/build/cmake/android.toolchain.cmake" ^
    -DCMAKE_MAKE_PROGRAM="%ANDROID_NDK%/prebuilt/windows-x86_64/bin/make.exe" ^
    -DANDROID_NDK=%ANDROID_NDK% ^
    -DANDROID_ABI=%ANDROID_ABI% ^
    -DANDROID_TOOLCHAIN_NAME=clang ^
    -DANDROID_NATIVE_API_LEVEL=21 ^
    -G "Unix Makefiles" ^
    %* ^
    || exit /b

cmake --build . || exit /b
