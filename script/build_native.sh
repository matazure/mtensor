mkdir -p build
cd build
cmake ..  $@

starttime=`date +'%Y-%m-%d %H:%M:%S'`

cmake --build . -- -j|| exit 1

#执行程序
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "build cost time： "$((end_seconds-start_seconds))"s"
