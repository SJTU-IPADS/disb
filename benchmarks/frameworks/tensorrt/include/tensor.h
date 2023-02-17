#ifndef _DISB_TENSORRT_TENSOR_H_
#define _DISB_TENSORRT_TENSOR_H_

#include "utils.h"

#include <memory>
#include <string.h>
#include <assert.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <jsoncpp/json/json.h>

nvinfer1::Dims convertDims(const Json::Value &tensorShape, const Json::Value &tensorShapeVariables);
struct Tensor
{
    int binding = -1;
    void *hostBuffer = nullptr;
    void *deviceBuffer = nullptr;
    size_t elementSize = 0;
    
    std::string name;
    Json::Value shape;
    nvinfer1::Dims dims;

    Tensor(const std::string &_name): name(_name) {}
    Tensor(const std::string &_name, const Json::Value &_shape): name(_name), shape(_shape) {}
    ~Tensor()
    {
        if (hostBuffer) free(hostBuffer);
        if (deviceBuffer) cudaFree(deviceBuffer);
    }

    void setDims(const Json::Value &shapeVar)
    {
        dims = convertDims(shape, shapeVar);
    }

    void copyToDeviceAsync(const cudaStream_t &stream) const
    {
        const size_t byteSize = elementSize * volume(dims);
        // printf("copy size %ld to device\n", byteSize);
        CHECK(cudaMemcpyAsync(deviceBuffer, hostBuffer, byteSize, cudaMemcpyHostToDevice, stream));
    }

    void copyToHostAsync(const cudaStream_t &stream) const
    {
        const size_t byteSize = elementSize * volume(dims);
        // printf("copy size %ld to host\n", byteSize);
        CHECK(cudaMemcpyAsync(hostBuffer, deviceBuffer, byteSize, cudaMemcpyDeviceToHost, stream));
    }
};

struct TensorBuffer
{
    std::vector<void *> deviceBuffers;

    TensorBuffer(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                int maxBatchSize, int profileIdx, const Json::Value &shapeVarMax,
                const std::vector<std::shared_ptr<Tensor>> &inputTensors,
                const std::vector<std::shared_ptr<Tensor>> &outputTensors);

    void createIOTensor(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                        std::shared_ptr<Tensor> tensor,
                        const std::string &bindingName,
                        const Json::Value &shapeVarMax);

    void *const *getDeviceBuffers() const
    {
        return deviceBuffers.data();
    }
};

#endif
