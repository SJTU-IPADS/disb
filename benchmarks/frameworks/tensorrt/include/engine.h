#ifndef _DISB_TENSORRT_ENGINE_H_
#define _DISB_TENSORRT_ENGINE_H_

#include <string>
#include <memory>
#include <NvInfer.h>
#include <jsoncpp/json/json.h>

#include "utils.h"

class TensorRTEngine
{
private:
    Logger logger;

public:
    Json::Value shapeVarMin;
    Json::Value shapeVarOpt;
    Json::Value shapeVarMax;

    int maxBatchSize = 1;
    bool enableExplicitBatchSize;

    std::shared_ptr<nvinfer1::ICudaEngine> engine;

public:
    TensorRTEngine(const Json::Value &modelConfig,
                   const std::string &modelDir,
                   int profileCnt, int64_t memoryLimit, cudaStream_t cudaStream);
};

#endif