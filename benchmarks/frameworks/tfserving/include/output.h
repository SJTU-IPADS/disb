#ifndef _DISB_TF_OUTPUT_H_
#define _DISB_TF_OUTPUT_H_

#include <vector>
#include <jsoncpp/json/json.h>

#include "tensorflow/core/framework/tensor.grpc.pb.h"

class OutputProcessor
{
public:
    virtual void processOutput(int batchSize, bool inferSuccess, const std::vector<int64_t> &selectIdxs,
                               const google::protobuf::Map<std::string, tensorflow::TensorProto> &outputTensors) = 0;
    virtual int checkCorrect() = 0;
};

class SoftmaxOutputProcessor: public OutputProcessor
{
private:
    bool outputComparable;
    int correctCnt = 0;
    int indexSize;
    std::vector<int> expectedOuputs;
    std::vector<std::string> outputTensorNames;
    float (*getLogit)(int idx, const tensorflow::TensorProto &tensor);

public:
    SoftmaxOutputProcessor(const Json::Value &modelConfig);

    virtual void processOutput(int batchSize, bool inferSuccess, const std::vector<int64_t> &selectIdxs,
                               const google::protobuf::Map<std::string, tensorflow::TensorProto> &outputTensors) override;
    virtual int checkCorrect() override;
};

class HiddenStateProcessor: public OutputProcessor
{
private:
    bool outputComparable;
    int correctCnt = 0;

public:
    HiddenStateProcessor(const Json::Value &modelConfig);

    virtual void processOutput(int batchSize, bool inferSuccess, const std::vector<int64_t> &selectIdxs,
                               const google::protobuf::Map<std::string, tensorflow::TensorProto> &outputTensors) override;
    virtual int checkCorrect() override;
};

#endif