mkdir -p build
cd build
cmake ..  $@

starttime=`date +'%Y-%m-%d %H:%M:%S'`

cmake --build . -- -j|| exit 1
