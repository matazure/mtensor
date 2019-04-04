FROM ubuntu:18.04
MAINTAINER p3.1415@qq.com
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y wget 
RUN apt-get install -y unzip
RUN apt-get install -y sshpass
RUN cd /
RUN wget https://dl.google.com/android/repository/android-ndk-r16b-linux-x86_64.zip && unzip -q android-ndk-r16b-linux-x86_64.zip && rm android-ndk-r16b-linux-x86_64.zip
ENV ANDROID_NDK=/android-ndk-r16b
