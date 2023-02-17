#ifndef _DISB_TRITON_OUTPUT_H_
#define _DISB_TRITON_OUTPUT_H_

#include <vector>
#include <jsoncpp/json/json.h>

#include "grpc_client.h"
#include "triton_utils.h"

class OutputProcessor
{
protected:
    std::vector<const triton::client::InferRequestedOutput *> tensors;

public:
    ~OutputProcessor() { for (auto tensor : tensors) delete tensor; }
    std::vector<const triton::client::InferRequestedOutput *> &outputTensors() { return tensors; }

    virtual void processOutput(int batchSize, bool inferSuccess,
                               const std::vector<int64_t> &selectIdxs,
                               std::shared_ptr<triton::client::InferResult> inferResult) = 0;
    virtual int checkCorrect() = 0;
};

class SoftmaxOutputProcessor: public OutputProcessor
{
private:
    bool outputComparable;
    int correctCnt = 0;
    int indexSize;
    std::string outputTensorName;
    std::vector<int> expectedOuputs;
    float (*getLogit)(int idx, const uint8_t *tensorBuffer);

public:
    SoftmaxOutputProcessor(const Json::Value &modelConfig,
                           const inference::ModelMetadataResponse &modelMeta);

    virtual void processOutput(int batchSize, bool inferSuccess,
                               const std::vector<int64_t> &selectIdxs,
                               std::shared_ptr<triton::client::InferResult> inferResult) override;
    virtual int checkCorrect() override;
};

class HiddenStateProcessor: public OutputProcessor
{
private:
    bool outputComparable;
    int correctCnt = 0;

public:
    HiddenStateProcessor(const Json::Value &modelConfig,
                         const inference::ModelMetadataResponse &modelMeta);

    virtual void processOutput(int batchSize, bool inferSuccess,
                               const std::vector<int64_t> &selectIdxs,
                               std::shared_ptr<triton::client::InferResult> inferResult) override;
    virtual int checkCorrect() override;
};

#endif