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
ENV ANDROID_NDK=/android-ndk-r16b

