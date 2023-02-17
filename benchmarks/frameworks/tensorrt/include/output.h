#ifndef _DISB_TENSORRT_OUTPUT_H_
#define _DISB_TENSORRT_OUTPUT_H_

#include <vector>
#include <jsoncpp/json/json.h>

#include "tensor.h"

class OutputProcessor
{
public:
    virtual void processOutput(bool inferSuccess,
                               const std::vector<std::shared_ptr<Tensor>> &outputTensors,
                               const std::vector<int64_t> &selectIdxs) = 0;
    virtual int checkCorrect() = 0;
};

class SoftmaxOutputProcessor: public OutputProcessor
{
private:
    bool outputComparable;
    int correctCnt = 0;
    std::vector<int> expectedOuputs;
    float (*getLogit)(int idx, void *buffer);

public:
    SoftmaxOutputProcessor(const Json::Value &modelConfig);

    virtual void processOutput(bool inferSuccess,
                               const std::vector<std::shared_ptr<Tensor>> &outputTensors,
                               const std::vector<int64_t> &selectIdxs) override;
    virtual int checkCorrect() override;
};

class HiddenStateProcessor: public OutputProcessor
{
private:
    bool outputComparable;
    int correctCnt = 0;

public:
    HiddenStateProcessor(const Json::Value &modelConfig);

    virtual void processOutput(bool inferSuccess,
                               const std::vector<std::shared_ptr<Tensor>> &outputTensors,
                               const std::vector<int64_t> &selectIdxs) override;
    virtual int checkCorrect() override;
};

#endif