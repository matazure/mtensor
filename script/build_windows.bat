if not exist build_win mkdir build_win
cd build_win

cmake .. -G "Visual Studio 15 2017 Win64" %*
cmake --build . --config Release -- /m || exit /b
REM exit /b 0
