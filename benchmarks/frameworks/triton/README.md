# Triton Benchmark for DISB
This is an DISB implementation of [Triton Inference Server](https://github.com/triton-inference-server).

Versions: Triton Inference Server 2.25.0, TensorRT 8.4.2.4, CUDA 11.7.1

Triton Benchmark chooses TensorRT as serving backend. The frontend client uses gRPC to issue infer requests to the triton backend.




## Dependencies

### Install NVIDIA Docker
```shell
# install nvidia-docker, in order to use gpus
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo systemctl restart docker

# pull serving triton server backend docker image
# Triton Inference Server 2.25.0, TensorRT 8.4.2.4, CUDA 11.7.1
# https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel_22-08.html#rel_22-08
docker pull nvcr.io/nvidia/tritonserver:22.08-py3
```

### Option 1: Use Pre-built Docker Image (Recommended)

```shell
docker pull shenwhang/disb-triton-client:0.2

# under disb/
docker run -it --name disb-triton-client --net=host -v ${PWD}:/workspace/disb shenwhang/disb-triton-client:0.2 /bin/bash
# or
make triton-front
```

### Option 2: Install Dependencies Manually

Use [Triton Client](https://github.com/triton-inference-server/client) docker image, or you can follow its guide to build locally.

```shell
# pull triton client image
docker pull nvcr.io/nvidia/tritonserver:22.08-py3-sdk

docker run -it --name disb-triton-client --net=host -v ${PWD}:/workspace/disb nvcr.io/nvidia/tritonserver:22.08-py3-sdk /bin/bash

# in disb-triton-client container
# install cmake, see https://cmake.org/download/
sh /workspace/disb/cmake-3.25.2-linux-x86_64.sh --prefix=/usr/local --exclude-subdir

# install dependencies
apt install rapidjson-dev jsoncpp-dev build-essential

# build triton client
cd /workspace/client
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/workspace/triton-client \
-DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON \
-DTRITON_COMMON_REPO_TAG=r22.08 -DTRITON_THIRD_PARTY_REPO_TAG=r22.08 \
-DTRITON_CORE_REPO_TAG=r22.08 -DTRITON_BACKEND_REPO_TAG=r22.08 ..
make cc-clients
# also copy third party dependencies
cp -r ./third-party /workspace/triton-client
```



## Convert ONNX to TensorRT Plan
```shell
# under disb/
docker run -it --name disb-trt8.4 --gpus all -v ${PWD}:/workspace/disb shenwhang/disb-trt8.4:0.1 /bin/bash
# or
make trt-container

# in docker container
cd /workspace/disb
tools/onnx_to_tensorrt.sh
exit
```



## Build

```shell
# under disb/ directory (or /workspace/disb/ if you choose pre-built docker image)
make triton
```



## Run

```shell
# start triton server backend (under disb/ in host)
make triton-back

# under disb/ directory (or /workspace/disb/ if you choose pre-built docker image)
# run full test
make triton-test

# run single test
# A means run DISB workload A
./run.sh triton A
./run.sh triton B
./run.sh triton C
./run.sh triton D
./run.sh triton E
./run.sh triton REAL
```
