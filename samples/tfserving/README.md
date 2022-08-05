# Tensorflow Serving Sample for DISB
## Dependencies

```shell
# install tools and libs
sudo apt install build-essential cmake autoconf libtool pkg-config
sudo apt install libopencv-dev libjsoncpp-dev

# build protobuf
sudo apt-get install automake curl make unzip
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure
make -j$(nproc)
make check # if you think this take too long, you can skip this step.
sudo make install
sudo ldconfig # refresh shared library cache.

# build grpc
git clone --recurse-submodules -b v1.46.3 --depth 1 --shallow-submodules https://github.com/grpc/grpc
cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j$(nproc)
sudo make install
popd

# install docker
sudo apt-get update
sudo apt install apt-transport-https ca-certificates curl software-properties-common && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && \
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable" && \
apt-cache policy docker-ce && \
sudo apt install -y containerd.io docker-ce docker-ce-cli && \
sudo systemctl status docker

# install nvidia-docker, in order to use gpus all
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo systemctl restart docker

# pull docker image
docker pull tensorflow/serving:latest-gpu
```



## Build

```shell
# under /disb directory
rm -rf build
cmake -DSAMPLE_TFSERVING=ON -B build
cd build
make -j$(nproc)
```



## Run

```shell
# under /disb directory
docker run -t --gpus all --rm -p 8500:8500 -p 8501:8501 -v $PWD/samples/tfserving/assets/resnet:/models/resnet -e MODEL_NAME=resnet tensorflow/serving:latest-gpu

# under /disb/build directory
./samples/tfserving/tf_benchmark ../samples/tfserving/assets/config/config.json
# you can change the config json file passed to tf_benchmark
```

