FROM ubuntu:18.04
MAINTAINER p3.1415@qq.com
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y clang
