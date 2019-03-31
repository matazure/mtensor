FROM nvidia/cuda:10.1-devel-ubuntu16.04
MAINTAINER p3.1415@qq.com
RUN apt-get update
RUN apt-get install -y cmake
# RUN apt-get install -y g++
# has g++5.4
