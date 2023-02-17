#include <limits>

#include "output.h"

float getFloat32(int idx, void *buffer)
{
    return ((float *)buffer)[idx];
}

SoftmaxOutputProcessor::SoftmaxOutputProcessor(const Json::Value &modelConfig)
{
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

void SoftmaxOutputProcessor::processOutput(bool inferSuccess,
                                           const std::vector<std::shared_ptr<Tensor>> &outputTensors,
                                           const std::vector<int64_t> &selectIdxs)
{
    correctCnt = 0;
    if (!inferSuccess) return;

    void *buffer = outputTensors[0]->hostBuffer;
    int batchSize = outputTensors[0]->dims.d[0];
    int outputSize = outputTensors[0]->dims.d[1];
    for (int i = 0; i < batchSize; ++i) {
        int outputIdx = 0;
        double maxLogit = std::numeric_limits<double>().min();
        for (int j = 0; j < outputSize; ++j) {
            double logit = getLogit(i * outputSize + j, buffer);
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

void HiddenStateProcessor::processOutput(bool inferSuccess,
                                         const std::vector<std::shared_ptr<Tensor>> &outputTensors,
                                         const std::vector<int64_t> &selectIdxs)
{
    correctCnt = 0;
    if (!inferSuccess) return;

    int batchSize = outputTensors[0]->dims.d[0];
    correctCnt = batchSize;
}

int HiddenStateProcessor::checkCorrect()
{
    return correctCnt;
}