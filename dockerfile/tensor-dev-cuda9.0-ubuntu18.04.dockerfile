FROM nvidia/cuda:9.0-devel-centos7
RUN yum install -y wget
RUN wget https://cmake.org/files/v3.12/cmake-3.12.0-Linux-x86_64.sh
RUN chmod +x cmake-3.12.0-Linux-x86_64.sh
RUN ./cmake-3.12.0-Linux-x86_64.sh --skip-license