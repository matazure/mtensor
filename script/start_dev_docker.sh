#!/bin/bash

docker run -ti --gpus all --privileged=true --network host $@  -v /root/.ssh:/root/.ssh -v $(pwd):/tensor -w /tensor tensor-dev:latest
