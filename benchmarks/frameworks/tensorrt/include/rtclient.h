#ifndef _DISB_TENSORRT_RTCLIENT_H_
#define _DISB_TENSORRT_RTCLIENT_H_

#include "disb.h"
#include "tensor.h"
#include "engine.h"
#include "input.h"
#include "output.h"

#include <map>
#include <mutex>
#include <string>
#include <memory>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

class TensorRTClient: public DISB::Client
{
private:
    std::string modelName;
    std::string modelDir;
    std::string benchmarksDir;
    Json::Value modelConfig;
    int64_t memoryLimit;

    int profileIdx;
    int profileCnt;
    int maxBatchSize;
    bool enableExplicitBatchSize;
    
    bool inferSuccess;
    int batchSize = 1;
    Json::Value shapeVar;
    std::vector<std::shared_ptr<Tensor>> inputTensors;
    std::vector<std::shared_ptr<Tensor>> outputTensors;
    std::shared_ptr<InputProcessor> inputProcessor;
    std::shared_ptr<OutputProcessor> outputProcessor;
    std::shared_ptr<std::vector<int64_t>> selectedIdxs;

    cudaStream_t cudaStream;
    std::unique_ptr<TensorBuffer> tensorBuffer;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    static std::mutex enginePoolMtx;
    static std::map<std::string, std::shared_ptr<TensorRTEngine>> enginePool;

public:
    TensorRTClient(const Json::Value &config);
    ~TensorRTClient();

    virtual void init() override;
    virtual void prepareInput() override;
    virtual void preprocess() override;
    virtual void copyInput() override;
    virtual void infer() override;
    virtual void copyOutput() override;
    virtual void postprocess() override;

    int getBatchSize() const;
    int getCorrectInferCount();

    static void freeEngines() { enginePool.clear(); }
};

std::shared_ptr<DISB::Client> rtClientFactory(const Json::Value &config);

#endif
