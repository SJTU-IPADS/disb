#include "output.h"
#include "triton_utils.h"

float getFloat32(int idx, const uint8_t *tensorBuffer)
{
    return *(float *)(tensorBuffer + idx * sizeof(float));
}

SoftmaxOutputProcessor::SoftmaxOutputProcessor(const Json::Value &modelConfig,
                                               const inference::ModelMetadataResponse &modelMeta)
{
    assert(modelMeta.outputs_size() == 1);
    outputTensorName = modelMeta.outputs(0).name();

    triton::client::InferRequestedOutput *output;
    ASSERT_TRITON_ERROR(triton::client::InferRequestedOutput::Create(&output, outputTensorName));
    tensors.emplace_back(output);

    indexSize = modelConfig["output_tensor_shapes"][0][1].asInt();
    outputComparable = modelConfig["output_comparable"].asBool();
    if (!outputComparable) return;

    std::string dataType = modelMeta.outputs(0).datatype();
    if (dataType == "FP32") {
        getLogit = getFloat32;
    } else {
        std::cout << "Error: Unknown output data type: " << dataType << std::endl;
        exit(-1);
    }

    for (const Json::Value &expectedOuput : modelConfig["outputs"]) {
        expectedOuputs.push_back(expectedOuput.asInt());
    }
}

void SoftmaxOutputProcessor::processOutput(int batchSize, bool inferSuccess,
                                           const std::vector<int64_t> &selectIdxs,
                                           std::shared_ptr<triton::client::InferResult> inferResult)
{
    correctCnt = 0;
    if (!inferSuccess) return;
    if (!outputComparable) { correctCnt = batchSize; return; }

    size_t bufferSize;
    const uint8_t *tensorBuffer;
    ASSERT_TRITON_ERROR(inferResult->RawData(outputTensorName, &tensorBuffer, &bufferSize));
    assert(bufferSize == indexSize * sizeof(float));

    for (int i = 0; i < batchSize; ++i) {
        int outputIdx = 0;
        double maxLogit = std::numeric_limits<double>().min();
        for (int j = 0; j < indexSize; ++j) {
            double logit = getLogit(i * indexSize + j, tensorBuffer);
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

HiddenStateProcessor::HiddenStateProcessor(const Json::Value &modelConfig,
                                           const inference::ModelMetadataResponse &modelMeta)
{
    for (int i = 0; i < modelMeta.outputs_size(); ++i) {
        triton::client::InferRequestedOutput *output;
        ASSERT_TRITON_ERROR(triton::client::InferRequestedOutput::Create(&output, modelMeta.outputs(i).name()));
        tensors.emplace_back(output);
    }

    outputComparable = modelConfig["output_comparable"].asBool();
}

void HiddenStateProcessor::processOutput(int batchSize, bool inferSuccess,
                                         const std::vector<int64_t> &selectIdxs,
                                         std::shared_ptr<triton::client::InferResult> inferResult)
{
    correctCnt = 0;
    if (!inferSuccess) return;
    if (!outputComparable) { correctCnt = batchSize; return; }

    std::cout << "Error: comparable HiddenState not implemented" << std::endl;
    exit(-1);
}

int HiddenStateProcessor::checkCorrect()
{
    return correctCnt;
}