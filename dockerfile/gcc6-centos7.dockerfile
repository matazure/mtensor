FROM centos:7
MAINTAINER p3.1415@qq.com
RUN yum update
RUN yum install -y centos-release-scl
RUN yum install -y devtoolset-6-gcc
scl enable devtoolset-6 bash
