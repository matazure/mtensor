FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER p3.1415@qq.com
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y g++-6
RUN rm /usr/bin/g++ /usr/bin/gcc
RUN ln -s /usr/bin/g++-6 /usr/bin/g++
RUN ln -s /usr/bin/gcc-6 /usr/bin/gcc
# RUN apt-get install -y g++
# has g++5.4
