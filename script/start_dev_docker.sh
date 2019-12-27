#!/bin/bash

docker run -ti $@ -v $(pwd):/tensor -w /tensor tensor-dev:latest
