if not exist build_win mkdir build_win
cd build_win
cmake .. -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=OFF
