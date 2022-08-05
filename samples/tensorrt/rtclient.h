#ifndef _DISB_TENSORRT_RTCLIENT_H_
#define _DISB_TENSORRT_RTCLIENT_H_

#include "disb.h"
#include "buffer.h"

#include <memory>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class TensorRTClient: public DISB::Client
{
public:
    TensorRTClient(const Json::Value &config);
    ~TensorRTClient() {}

    virtual void init() override;
    virtual void prepareInput() override;
    virtual void preprocess() override;
    virtual void copyInput() override;
    virtual void infer() override;
    virtual void copyOutput() override;
    virtual void postprocess() override;

    int getInputNumber();
    int getOutputNumber();

private:
    std::string modelName;
    std::string modelPath;
    std::string inputDataPath;

    int inputH;
    int inputW;
    int outputSize;

    int inputNumber;
    int outputNumber;
    cv::Mat inputImage;

    Logger logger;
    std::shared_ptr<ICudaEngine> mEngine;
    std::unique_ptr<BufferManager> buffers;
    std::unique_ptr<IExecutionContext> context;
};

std::shared_ptr<DISB::Client> rtClientFactory(const Json::Value &config);

#endif
