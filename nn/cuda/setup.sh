#!/bin/bash
#
# Sets environment variables needed for working with nn/cuda.
# Designed to be executed from the root of the GOPATH. Install with
#
#		cd ml
#		ln -s src/github.com/crawshaw/ml/nn/cuda/setup.sh .
#
# Then when beginning a session
#
#		cd ml
#		source setup.sh
#

if [ ! -x $(pwd)/src/github.com/crawshaw/ml ]; then
	echo "does not look like the root of a GOPATH"
	exit 1
fi

export GOPATH=$(pwd)

if [ -x /Developer/NVIDIA/CUDA-5.5 ]; then
	CUDAROOT=/Developer/NVIDIA/CUDA-5.5
fi
if [ -x /usr/local/cuda-5.5 ]; then
	CUDAROOT=/usr/local/cuda-5.5
fi
export CUDAROOT

if [[ -x $CUDAROOT/lib64 && $(uname -m) == "x86_64" ]]; then
	CUDALIB=$CUDAROOT/lib64
else
	CUDALIB=$CUDAROOT/lib
fi
export CUDALIB

export CGO_LDFLAGS=-L$CUDALIB

LD_LIBRARY_PATH=$(pwd)/src/github.com/crawshaw/ml/nn/cuda
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDALIB
export LD_LIBRARY_PATH

if [ ! `which nvcc` ]; then
	export PATH=$PATH:/usr/local/cuda-5.5/bin
fi
