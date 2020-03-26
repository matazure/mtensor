#!/bin/bash

docker run -ti --runtime=nvidia --privileged=true $@  -v /root/.ssh:/root/.ssh -v $(pwd):/tensor -w /tensor tensor-dev:latest
