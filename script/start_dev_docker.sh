#!/bin/bash

docker run -ti --runtime=nvidia --privileged=true $@ -v $(pwd):/tensor -w /tensor tensor-dev:latest
