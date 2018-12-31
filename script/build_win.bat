if not exist build_win mkdir build_win
cd build_win

cmake .. -G "Visual Studio 14 2015 Win64" %*
cmake --build . --config Release || exit /b
REM exit /b 0
