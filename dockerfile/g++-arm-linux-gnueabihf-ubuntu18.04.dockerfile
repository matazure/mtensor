FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y g++-arm-linux-gnueabihf
