#ifndef _DISB_TRITON_TRITONCLIENT_H_
#define _DISB_TRITON_TRITONCLIENT_H_

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include "grpc_client.h"

#include "disb.h"
#include "input.h"
#include "output.h"

class TritonClient: public DISB::Client
{
public:
    TritonClient(const Json::Value &config);

    virtual void init() override;
    virtual void prepareInput() override;
    virtual void preprocess() override;
    virtual void infer() override;
    virtual void postprocess() override;

    int getBatchSize() const;
    int getCorrectInferCount();

private:
    std::string modelName;
    std::string modelDir;
    std::string benchmarksDir;
    Json::Value modelConfig;
    
    int batchSize;
    Json::Value shapeVar;
    std::shared_ptr<InputProcessor> inputProcessor;
    std::shared_ptr<OutputProcessor> outputProcessor;
    std::shared_ptr<std::vector<int64_t>> selectedIdxs;

    std::shared_ptr<triton::client::InferResult> inferResult;
    std::unique_ptr<triton::client::InferOptions> inferOptions;
    std::unique_ptr<triton::client::InferenceServerGrpcClient> grpcClient;
};

std::shared_ptr<DISB::Client> tritonClientFactory(const Json::Value &config);

#endif