FROM ubuntu:18.04
MAINTAINER p3.1415@qq.com
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y wget 
RUN wget https://publishing-ap-linaro-org.s3.amazonaws.com/releases/components/toolchain/binaries/latest-7/arm-linux-gnueabihf/gcc-linaro-7.4.1-2019.02-x86_64_arm-linux-gnueabihf.tar.xz?Signature=aOPpW2h5zSYSsANK96U27iPUc2Q%3D&Expires=1554097422&AWSAccessKeyId=AKIAIELXV2RYNAHFUP7A -O gcc-linaro-7.4.1-2019.02-x86_64_arm-linux-gnueabihf.tar
