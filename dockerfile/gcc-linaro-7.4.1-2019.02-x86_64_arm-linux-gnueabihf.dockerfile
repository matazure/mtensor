FROM ubuntu:18.04
MAINTAINER p3.1415@qq.com
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y wget 
ADD gcc-linaro-7.4.1-2019.02-x86_64_arm-linux-gnueabihf.tar / 
ENV GCC_LINARO_TOOLCHAIN=gcc-linaro-7.4.1-2019.02-x86_64_arm-linux-gnueabihf

