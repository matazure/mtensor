FROM nvidia/cuda:10.0-devel-ubuntu18.04
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y g++
RUN apt-get install -y g++-aarch64-linux-gnu
RUN apt-get install -y g++-arm-linux-gnueabihf
RUN apt-get install -y clang
RUN apt-get install -y wget
RUN apt-get install -y unzip
RUN wget https://dl.google.com/android/repository/android-ndk-r16b-linux-x86_64.zip && unzip -q android-ndk-r16b-linux-x86_64.zip && rm android-ndk-r16b-linux-x86_64.zip
RUN apt-get install -y vim
RUN apt-get install -y gdb
RUN apt-get install -y git
RUN apt-get install -y software-properties-common
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN add-apt-repository "deb http://apt.llvm.org/bionic/   llvm-toolchain-bionic-9  main"
RUN apt-get update 
RUN apt-get install -y clang-9 lldb-9 lld-9 clangd-9
RUN apt-get install -y libomp5
RUN apt-get install -y libomp-dev
ENV ANDROID_NDK=/android-ndk-r16b

