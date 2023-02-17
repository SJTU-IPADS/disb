#ifndef _DISB_TENSORRT_INPUT_H_
#define _DISB_TENSORRT_INPUT_H_

#include <vector>
#include <memory>
#include "tensor.h"
#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

class InputProcessor
{
public:
    virtual void prepare(int batchSize) = 0;
    virtual std::shared_ptr<std::vector<int64_t>> preprocess(std::vector<std::shared_ptr<Tensor>> &inputTensors,
                                                             Json::Value &shapeVar) = 0;
};

class CommonImageInputProcessor: public InputProcessor
{
private:
    int batchSize;
    cv::Size imgSize;
    size_t singleImageSize;
    std::vector<cv::Mat> availableImages;
    std::shared_ptr<std::vector<int64_t>> selectedIdxs;

public:
    CommonImageInputProcessor(const std::string &modelDir, const Json::Value &modelConfig);
    virtual void prepare(int batchSize) override;
    virtual std::shared_ptr<std::vector<int64_t>> preprocess(std::vector<std::shared_ptr<Tensor>> &inputTensors,
                                                             Json::Value &shapeVar) override;
};

class VocabAttentionInputProcessor: public InputProcessor
{
private:
    int batchSize;
    size_t maxSeqLength;
    std::vector<std::vector<int32_t>> availableVocabIds;
    std::vector<std::vector<int32_t>> availableAttentionMasks;
    std::shared_ptr<std::vector<int64_t>> selectedIdxs;

public:
    VocabAttentionInputProcessor(const Json::Value &modelConfig);
    virtual void prepare(int batchSize) override;
    virtual std::shared_ptr<std::vector<int64_t>> preprocess(std::vector<std::shared_ptr<Tensor>> &inputTensors,
                                                             Json::Value &shapeVar) override;
};

#endif