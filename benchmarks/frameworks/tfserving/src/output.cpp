#include "output.h"

float getFloat32(int idx, const tensorflow::TensorProto &tensor)
{
    return tensor.float_val(idx);
}

SoftmaxOutputProcessor::SoftmaxOutputProcessor(const Json::Value &modelConfig)
{
    for (const auto &outputTensorName : modelConfig["output_tensor_names_tf"]) {
        outputTensorNames.push_back(outputTensorName.asString());
    }

    indexSize = modelConfig["output_tensor_shapes"][0][1].asInt();
    outputComparable = modelConfig["output_comparable"].asBool();
    if (!outputComparable) return;

    std::string dataType = modelConfig["output_tensor_types"][0].asString();
    if (dataType == "float32") {
        getLogit = getFloat32;
    }

    for (const Json::Value &expectedOuput : modelConfig["outputs"]) {
        expectedOuputs.push_back(expectedOuput.asInt());
    }
}

void SoftmaxOutputProcessor::processOutput(int batchSize, bool inferSuccess, const std::vector<int64_t> &selectIdxs,
                                           const google::protobuf::Map<std::string, tensorflow::TensorProto> &outputTensors)
{
    correctCnt = 0;
    if (!inferSuccess) return;

    auto &tensor = outputTensors.find(outputTensorNames[0])->second;
    for (int i = 0; i < batchSize; ++i) {
        int outputIdx = 0;
        double maxLogit = std::numeric_limits<double>().min();
        for (int j = 0; j < indexSize; ++j) {
            double logit = getLogit(i * indexSize + j, tensor);
            // printf("tensor[%d] = %.2f\n", i * outputSize + j, logit);
            if (logit > maxLogit) {
                maxLogit = logit;
                outputIdx = j;
            }
        }
        // printf("batch: %d, output: %d\n", i, outputIdx);
        correctCnt += (outputIdx == expectedOuputs[selectIdxs[i]]) || !outputComparable;
    }
}

int SoftmaxOutputProcessor::checkCorrect()
{
    return correctCnt;
}

HiddenStateProcessor::HiddenStateProcessor(const Json::Value &modelConfig)
{
    outputComparable = modelConfig["output_comparable"].asBool();
    if (!outputComparable) return;
}

void HiddenStateProcessor::processOutput(int batchSize, bool inferSuccess, const std::vector<int64_t> &selectIdxs,
                                         const google::protobuf::Map<std::string, tensorflow::TensorProto> &outputTensors)
{
    correctCnt = 0;
    if (!inferSuccess) return;

    correctCnt = batchSize;
}

int HiddenStateProcessor::checkCorrect()
{
    return correctCnt;
}