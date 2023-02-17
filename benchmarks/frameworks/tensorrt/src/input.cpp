#include <stdlib.h>

#include "disb_utils.h"
#include "input.h"

CommonImageInputProcessor::CommonImageInputProcessor(const std::string &modelDir, const Json::Value &modelConfig)
    : selectedIdxs(std::make_shared<std::vector<int64_t>>())
{
    std::string inputDir = joinPath(modelDir, modelConfig["input_dir"].asString());
    imgSize = cv::Size(modelConfig["input_tensor_shapes"][0][1].asInt(), modelConfig["input_tensor_shapes"][0][2].asInt());
    singleImageSize = 3 * imgSize.width * imgSize.height;
    
    for (const auto &inputFilename : modelConfig["inputs"]) {
        cv::Mat inputImage = cv::imread(joinPath(inputDir, inputFilename.asString()));
        cv::resize(inputImage, inputImage, imgSize);
        cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
        inputImage.convertTo(inputImage, CV_32FC3, -1 / 255.0, 1);
        availableImages.push_back(inputImage);
    }

    srand(unsigned(time(nullptr)));
}

void CommonImageInputProcessor::prepare(int _batchSize)
{
    batchSize = _batchSize;
    selectedIdxs->clear();

    for (int i = 0; i < batchSize; ++i) {
        int idx = rand() % availableImages.size();
        selectedIdxs->push_back(idx);
    }
}

std::shared_ptr<std::vector<int64_t>>
CommonImageInputProcessor::preprocess(std::vector<std::shared_ptr<Tensor>> &inputTensors,
                                      Json::Value &shapeVar)
{
    shapeVar["batch_size"] = batchSize;
    // Read the input data into the managed buffers
    float* hostDataBuffer = static_cast<float*>(inputTensors[0]->hostBuffer);
    int pixelCnt = imgSize.area();
    for (int i = 0; i < batchSize; ++i) {
        cv::Mat &inputImage = availableImages[(*selectedIdxs)[i]];
        memcpy(hostDataBuffer + i * singleImageSize, inputImage.data, singleImageSize * sizeof(float));
    }
    return selectedIdxs;
}

VocabAttentionInputProcessor::VocabAttentionInputProcessor(const Json::Value &modelConfig)
    : selectedIdxs(std::make_shared<std::vector<int64_t>>())
{
    for (const auto &sequence : modelConfig["inputs"]) {
        availableVocabIds.emplace_back();
        availableAttentionMasks.emplace_back();
        const auto &vocabIds = sequence[0];
        const auto &attentionMasks = sequence[1];
        int seqLen = vocabIds.size();
        for (int i = 0; i < seqLen; ++i) {
            availableVocabIds.back().push_back(vocabIds[i].asInt());
            availableAttentionMasks.back().push_back(attentionMasks[i].asInt());
        }
    }
}

void VocabAttentionInputProcessor::prepare(int _batchSize)
{
    maxSeqLength = 0;
    batchSize = _batchSize;
    selectedIdxs->clear();

    for (int i = 0; i < batchSize; ++i) {
        int idx = rand() % availableVocabIds.size();
        if (availableVocabIds[idx].size() > maxSeqLength) {
            maxSeqLength = availableVocabIds[idx].size();
        }
        selectedIdxs->push_back(idx);
    }
}

std::shared_ptr<std::vector<int64_t>>
VocabAttentionInputProcessor::preprocess(std::vector<std::shared_ptr<Tensor>> &inputTensors,
                                         Json::Value &shapeVar)
{
    shapeVar["batch_size"] = batchSize;
    shapeVar["sequence"] = (Json::Value::Int)maxSeqLength;

    int32_t* vocabIdBuffer = static_cast<int32_t*>(inputTensors[0]->hostBuffer);
    int32_t* attentionMaskBuffer = static_cast<int32_t*>(inputTensors[1]->hostBuffer);
    for (int i = 0; i < batchSize; ++i) {
        int64_t idx = (*selectedIdxs)[i];
        auto &vocabIds = availableVocabIds[idx];
        auto &attentionMask = availableAttentionMasks[idx];
        memcpy(vocabIdBuffer + i * maxSeqLength, vocabIds.data(), vocabIds.size() * sizeof(int32_t));
        memcpy(attentionMaskBuffer + i * maxSeqLength, attentionMask.data(), attentionMask.size() * sizeof(int32_t));
    }

    return selectedIdxs;
}