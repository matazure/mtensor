FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER p3.1415@qq.com
RUN apt-get update
RUN apt-get install -y cmake
