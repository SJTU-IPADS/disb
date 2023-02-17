#ifndef _DISB_TRITON_INPUT_H_
#define _DISB_TRITON_INPUT_H_

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include "grpc_client.h"

class InputProcessor
{
protected:
    std::vector<triton::client::InferInput *> tensors;

public:
    ~InputProcessor() { for (auto tensor : tensors) delete tensor; }
    std::vector<triton::client::InferInput *> &inputTensors() { return tensors; }

    virtual void prepare(int batchSize) = 0;

    /// @return selected indexes
    virtual std::shared_ptr<std::vector<int64_t>> preprocess(Json::Value &shapeVar) = 0;
};

class CommonImageInputProcessor: public InputProcessor
{
private:
    int batchSize;
    cv::Size imgSize;
    std::vector<std::vector<uchar>> availableImages;
    std::shared_ptr<std::vector<int64_t>> selectedIdxs;

public:
    CommonImageInputProcessor(const std::string &modelDir,
                              const Json::Value &modelConfig,
                              const inference::ModelMetadataResponse &modelMeta);
    virtual void prepare(int batchSize) override;
    virtual std::shared_ptr<std::vector<int64_t>> preprocess(Json::Value &shapeVar) override;
};

class VocabAttentionInputProcessor: public InputProcessor
{
private:
    int batchSize;
    size_t elementSize;
    size_t maxSeqLength;
    std::vector<std::vector<uchar>> emptyBufferPool;
    std::vector<std::vector<uchar>> availableVocabIds;
    std::vector<std::vector<uchar>> availableAttentionMasks;
    std::shared_ptr<std::vector<int64_t>> selectedIdxs;

public:
    VocabAttentionInputProcessor(const Json::Value &modelConfig,
                                 const inference::ModelMetadataResponse &modelMeta);
    virtual void prepare(int batchSize) override;
    virtual std::shared_ptr<std::vector<int64_t>> preprocess(Json::Value &shapeVar) override;
};

#endif