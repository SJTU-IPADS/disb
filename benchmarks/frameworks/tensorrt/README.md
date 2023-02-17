# TensorRT Benchmark for DISB

This is an DISB implementation of [TensorRT](https://github.com/NVIDIA/TensorRT).

Versions: TensorRT 8.4.2.4, CUDA 11.7.1, cuBLAS 11.10.3.66, cuDNN 8.5.0.96

TensorRT Benchmark is a single-process benchmark. It directly send infer requests to GPU and does not need RPC to communicate with the serving backend.



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
```



### Option 1: Use Pre-built Docker Image (Recommended)

```shell
docker pull shenwhang/disb-trt8.4:0.1

# under disb/
docker run -it --name disb-trt8.4 --gpus all -v ${PWD}:/workspace/disb shenwhang/disb-trt8.4:0.1 /bin/bash
# or
make trt-container
```



### Option 2: Install Dependencies Manually

```shell
# TensorRT 8.4.2.4, CUDA 11.7.1, cuBLAS 11.10.3.66, cuDNN 8.5.0.96
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_22-08.html#rel_22-08
docker pull nvcr.io/nvidia/tensorrt:22.08-py3

# under disb/
docker run -it --name disb-trt8.4 --gpus all -v ${PWD}:/workspace/disb nvcr.io/nvidia/tensorrt:22.08-py3 /bin/bash

# in docker container
apt update
apt install libopencv-dev libjsoncpp-dev
```



## Build

```shell
# under /workspace/disb/ in docker container
make trt
```



## Run

```shell
# under /workspace/disb/ in docker
# run full test
make trt-test

# run single test
# A means run DISB workload A
./run.sh trt A
./run.sh trt B
./run.sh trt C
./run.sh trt D
./run.sh trt E
./run.sh trt REAL
```
