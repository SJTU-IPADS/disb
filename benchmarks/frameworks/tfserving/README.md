# Tensorflow Serving Benchmark for DISB
This is an DISB implementation of [TensorFlow Serving](https://github.com/tensorflow/serving).

Versions: TensorFlow Serving 2.5.4-gpu

Tensorflow Serving Benchmark uses TensorFlow as serving backend. The frontend client uses gRPC to issue infer requests to the tensorflow serving backend.



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

# pull serving backend docker image
docker pull tensorflow/serving:2.5.4-gpu
```

### Option 1: Use Pre-built Docker Image (Recommended)

```shell
docker pull shenwhang/disb-tfs-client:1.0

# under disb/
docker run -it --name disb-tfs-client --network host -v ${PWD}:/workspace/disb shenwhang/disb-tfs-client:1.0 /bin/bash
# or
make tfs-front
```

### Option 2: Install Dependencies Manually
```shell
# install tools and libs
sudo apt install build-essential cmake autoconf libtool pkg-config vim
sudo apt install libopencv-dev libjsoncpp-dev

# build grpc and protobuf
git clone --recurse-submodules -b v1.48.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      ../..
make -j$(nproc)
sudo make install
popd
```



## Build

```shell
# under disb/ directory (or /workspace/disb/ if you choose pre-built docker image)
make tfs
```



## Run

```shell
# start tensorflow serving backend (under disb/ in host)
make tfs-back

# under disb/ directory (or /workspace/disb/ if you choose pre-built docker image)
# run full test
make tfs-test

# run single test
# A means run DISB workload A
./run.sh tfs A
./run.sh tfs B
./run.sh tfs C
./run.sh tfs D
./run.sh tfs E
./run.sh tfs REAL
```
