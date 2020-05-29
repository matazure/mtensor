FROM centos:7
RUN yum update -y
RUN yum install -y gcc
RUN yum install -y make
RUN yum install -y wget
RUN wget https://cmake.org/files/v3.10/cmake-3.10.0-Linux-x86_64.sh 
RUN chmod +x cmake-3.10.0-Linux-x86_64.sh
RUN mkdir /cmake && ./cmake-3.10.0-Linux-x86_64.sh --skip-license --prefix=/cmake && ln -s /cmake/bin/cmake /usr/bin/cmake
RUN yum install -y gcc-c++
