# TensorRT Sample for DISB

## Dependencies

### Docker Environment (recommended)
```shell
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

# build docker image
docker build -t tensorrt -f ./Dockerfile .

# if you do not want to build by yourself, you can pull pre-built image
docker pull shenwhang/tensorrt:v4

# run docker env
docker run -it --name tensorrt --gpus all -v ${PWD}:/workspace/disb tensorrt /bin/bash
# or
docker run -it --name tensorrt --gpus all -v ${PWD}:/workspace/disb shenwhang/tensorrt:v4 /bin/bash
```

### Loacl Environment (Ubuntu 20.04)
```shell
# install tools and libs
sudo apt install build-essential cmake
sudo apt install libopencv-dev libjsoncpp-dev

# add nvidia apt repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

# install cuda
sudo apt-get -y install cuda=11.4.4-1

# add cuda to PATH in .bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# install cudnn
sudo apt-get install libcudnn8=8.2.4.15-1+cuda11.4 libcudnn8-dev=8.2.4.15-1+cuda11.4

# install tensorrt
version="8.2.5-1+cuda11.4"
sudo apt-get install libnvinfer8=${version} libnvonnxparsers8=${version} libnvparsers8=${version} libnvinfer-plugin8=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version}
```



## Build

```shell
# under /disb directory
rm -rf build
cmake -DSAMPLE_TENSORRT=ON -B build
cd build
make
```



## Run

```shell
/workspace/disb/build/samples/tensorrt/tensorrt_benchmark /workspace/disb/samples/tensorrt/assets/config.json
```