FROM nvidia/cuda:9.0-devel-ubuntu16.04
RUN apt-get update -y
RUN apt-get install -y wget
RUN apt-get install -y make
RUN wget https://cmake.org/files/v3.10/cmake-3.10.0-Linux-x86_64.sh 
RUN chmod +x cmake-3.10.0-Linux-x86_64.sh
RUN mkdir /cmake && ./cmake-3.10.0-Linux-x86_64.sh --skip-license --prefix=/cmake && ln -s /cmake/bin/cmake /usr/bin/cmake
