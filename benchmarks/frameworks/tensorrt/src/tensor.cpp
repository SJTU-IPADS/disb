#include "tensor.h"

#include <set>

nvinfer1::Dims convertDims(const Json::Value &tensorShape, const Json::Value &tensorShapeVariables)
{
    nvinfer1::Dims dims;
    dims.nbDims = tensorShape.size();
    for (int i = 0; i < dims.nbDims; ++i) {
        if (tensorShape[i].isString()) {
            dims.d[i] = tensorShapeVariables[tensorShape[i].asString()].asInt();
        } else {
            dims.d[i] = tensorShape[i].asInt();
        }
    }
    return dims;
}

TensorBuffer::TensorBuffer(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                           int maxBatchSize, int profileIdx, const Json::Value &shapeVarMax,
                           const std::vector<std::shared_ptr<Tensor>> &inputTensors,
                           const std::vector<std::shared_ptr<Tensor>> &outputTensors)
{
    std::string tensorNameSuffix;
    if (profileIdx > 0) {
        tensorNameSuffix = std::string(" [profile ") + std::to_string(profileIdx) + "]";
    }

    std::map<std::string, std::shared_ptr<Tensor>> ioTensors;

    // tensors appears in model config as input or output tensors
    // the max size of these tensors are specified in config
    for (const auto &inputTensor : inputTensors) {
        std::string bindingName = inputTensor->name + tensorNameSuffix;
        createIOTensor(engine, inputTensor, bindingName, shapeVarMax);
        ioTensors[bindingName] = inputTensor;
    }

    for (const auto &outputTensor: outputTensors) {
        std::string bindingName = outputTensor->name + tensorNameSuffix;
        createIOTensor(engine, outputTensor, bindingName, shapeVarMax);
        ioTensors[bindingName] = outputTensor;
    }

    int nbBindings = engine->getNbBindings();
    
    for (int i = 0; i < nbBindings; ++i) {
        std::string tensorName(engine->getBindingName(i));
        
        auto it = ioTensors.find(tensorName);
        if (it != ioTensors.end()) {
            deviceBuffers.push_back(it->second->deviceBuffer);
        } else {
            deviceBuffers.push_back(nullptr);
        }
    }
}

void TensorBuffer::createIOTensor(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                                  std::shared_ptr<Tensor> tensor,
                                  const std::string &bindingName,
                                  const Json::Value &shapeVarMax)
{
    int binding = engine->getBindingIndex(bindingName.c_str());
    nvinfer1::Dims dims = convertDims(tensor->shape, shapeVarMax);
    size_t tensorSize = getElementSize(engine->getBindingDataType(binding));
    tensor->elementSize = tensorSize;
        
    int vecDim = engine->getBindingVectorizedDim(binding);
    if (-1 != vecDim) {
        // i.e., 0 != lgScalarsPerVector
        int scalarsPerVec = engine->getBindingComponentsPerElement(binding);
        dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
        tensorSize *= scalarsPerVec;
    }
    tensorSize *= volume(dims);

    // printf("buffer %s element size: %ld, size: %ld\n", tensor->name.c_str(), tensor->elementSize, tensorSize);

    void *hostBuffer = malloc(tensorSize);
    memset(hostBuffer, 0, tensorSize);
    if (!hostBuffer) throw std::bad_alloc();

    void *deviceBuffer = nullptr;
    if (cudaMalloc(&deviceBuffer, tensorSize) != cudaSuccess) throw std::bad_alloc();
    cudaMemset(deviceBuffer, 0, tensorSize);
    
    tensor->binding = binding;
    tensor->hostBuffer = hostBuffer;
    tensor->deviceBuffer = deviceBuffer;
}
