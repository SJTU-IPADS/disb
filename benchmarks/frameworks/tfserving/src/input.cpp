#include <stdlib.h>

#include "disb_utils.h"
#include "input.h"

CommonImageInputProcessor::CommonImageInputProcessor(const std::string &modelDir, const Json::Value &modelConfig)
    : selectedIdxs(std::make_shared<std::vector<int64_t>>())
{
    for (const auto &inputTensorName : modelConfig["input_tensor_names_tf"]) {
        inputTensorNames.push_back(inputTensorName.asString());
    }

    std::string inputDir = joinPath(modelDir, modelConfig["input_dir"].asString());
    imgSize = cv::Size(modelConfig["input_tensor_shapes"][0][1].asInt(), modelConfig["input_tensor_shapes"][0][2].asInt());

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
CommonImageInputProcessor::preprocess(google::protobuf::Map<std::string, tensorflow::TensorProto> &inputTensors, Json::Value &shapeVar)
{
    inputTensors.clear();
    shapeVar["batch_size"] = batchSize;

    tensorflow::TensorProto &tensor = inputTensors[inputTensorNames[0]];
    tensor.set_dtype(tensorflow::DataType::DT_FLOAT);

    for (int i = 0; i < batchSize; ++i) {
        cv::Mat &inputImage = availableImages[(*selectedIdxs)[i]];
        for (int row = 0; row < imgSize.height; ++row) {
            for (int col = 0; col < imgSize.width; ++col) {
                auto &rgb = inputImage.at<cv::Vec3f>(row, col);
                tensor.add_float_val(rgb[0]);
                tensor.add_float_val(rgb[1]);
                tensor.add_float_val(rgb[2]);
            }
        }
    }

    tensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
    tensor.mutable_tensor_shape()->add_dim()->set_size(imgSize.height);
    tensor.mutable_tensor_shape()->add_dim()->set_size(imgSize.width);
    tensor.mutable_tensor_shape()->add_dim()->set_size(3);

    return selectedIdxs;
}

VocabAttentionInputProcessor::VocabAttentionInputProcessor(const Json::Value &modelConfig)
    : selectedIdxs(std::make_shared<std::vector<int64_t>>())
{
    for (const auto &inputTensorName : modelConfig["input_tensor_names_tf"]) {
        inputTensorNames.push_back(inputTensorName.asString());
    }

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
VocabAttentionInputProcessor::preprocess(google::protobuf::Map<std::string, tensorflow::TensorProto>& inputTensors, Json::Value &shapeVar)
{
    shapeVar["batch_size"] = batchSize;
    shapeVar["sequence"] = (Json::Value::Int)maxSeqLength;

    tensorflow::TensorProto &vocabIdsTensor = inputTensors[inputTensorNames[0]];
    tensorflow::TensorProto &attentionMaskTensor = inputTensors[inputTensorNames[1]];
    vocabIdsTensor.set_dtype(tensorflow::DataType::DT_INT64);
    attentionMaskTensor.set_dtype(tensorflow::DataType::DT_INT64);

    for (int i = 0; i < batchSize; ++i) {
        int64_t idx = (*selectedIdxs)[i];
        auto &vocabIds = availableVocabIds[idx];
        auto &attentionMask = availableAttentionMasks[idx];

        int j = 0;
        for (; j < vocabIds.size(); ++j) {
            vocabIdsTensor.add_int64_val(vocabIds[j]);
            attentionMaskTensor.add_int64_val(attentionMask[j]);
        }
        for (; j < maxSeqLength; ++j) {
            vocabIdsTensor.add_int64_val(0);
            attentionMaskTensor.add_int64_val(0);
        }
    }

    vocabIdsTensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
    vocabIdsTensor.mutable_tensor_shape()->add_dim()->set_size(maxSeqLength);

    attentionMaskTensor.mutable_tensor_shape()->add_dim()->set_size(batchSize);
    attentionMaskTensor.mutable_tensor_shape()->add_dim()->set_size(maxSeqLength);

    return selectedIdxs;
}