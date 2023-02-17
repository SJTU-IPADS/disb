#include <vector>
#include <cstdlib>

#include "disb_utils.h"
#include "input.h"
#include "triton_utils.h"

namespace tc = triton::client;

CommonImageInputProcessor::CommonImageInputProcessor(const std::string &modelDir,
                                                     const Json::Value &modelConfig,
                                                     const inference::ModelMetadataResponse &modelMeta)
    : selectedIdxs(std::make_shared<std::vector<int64_t>>())
{
    assert(modelMeta.inputs_size() == 1);
    std::string dataType = modelMeta.inputs(0).datatype();
    assert(dataType == "FP32");

    std::string inputDir = joinPath(modelDir, modelConfig["input_dir"].asString());
    imgSize = cv::Size(modelConfig["input_tensor_shapes"][0][1].asInt(), modelConfig["input_tensor_shapes"][0][2].asInt());
    
    std::vector<int64_t> shape = { 1, imgSize.height, imgSize.width, 3 };
    tc::InferInput *input;
    ASSERT_TRITON_ERROR(tc::InferInput::Create(&input, modelMeta.inputs(0).name(), shape, dataType))
    tensors.push_back(input);

    for (const auto &inputFilename : modelConfig["inputs"]) {
        cv::Mat inputImage = cv::imread(joinPath(inputDir, inputFilename.asString()));
        cv::resize(inputImage, inputImage, imgSize);
        cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
        inputImage.convertTo(inputImage, CV_32FC3, -1 / 255.0, 1);

        std::vector<uchar> img(3 * imgSize.width * imgSize.height * sizeof(float));
        memcpy(img.data(), inputImage.data, 3 * imgSize.width * imgSize.height * sizeof(float));
        availableImages.push_back(img);
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

std::shared_ptr<std::vector<int64_t>> CommonImageInputProcessor::preprocess(Json::Value &shapeVar)
{
    shapeVar["batch_size"] = batchSize;

    tc::InferInput *tensor = tensors[0];
    tensor->SetShape({ batchSize, imgSize.height, imgSize.width, 3 });
    tensor->Reset();

    for (int i = 0; i < batchSize; ++i) {
        ASSERT_TRITON_ERROR(tensor->AppendRaw(availableImages[(*selectedIdxs)[i]]));
    }

    return selectedIdxs;
}

VocabAttentionInputProcessor::VocabAttentionInputProcessor(const Json::Value &modelConfig,
                                                           const inference::ModelMetadataResponse &modelMeta)
    : selectedIdxs(std::make_shared<std::vector<int64_t>>())
{
    assert(modelMeta.inputs_size() == 2);
    std::string vocabIdsTensorName = modelConfig["input_tensor_names_onnx"][0].asString();
    std::string attentionMasksTensorName = modelConfig["input_tensor_names_onnx"][1].asString();

    std::string dataType = modelMeta.inputs(0).datatype();
    assert(dataType == modelMeta.inputs(1).datatype());

    tensors.resize(2);
    std::vector<int64_t> shape = { 1, 1 };
    for (int i = 0; i < 2; ++i) {
        std::string tensorName = modelMeta.inputs(i).name();
        
        tc::InferInput *input;
        ASSERT_TRITON_ERROR(tc::InferInput::Create(&input, tensorName, shape, dataType))
        
        if (tensorName == vocabIdsTensorName) tensors[0] = input;
        else if (tensorName == attentionMasksTensorName) tensors[1] = input;
        else assert(0);
    }

    void (*setElement)(std::vector<uchar> &vector, int64_t idx, int64_t val);
    if (dataType == "INT32") {
        elementSize = sizeof(int32_t);
        setElement = [](std::vector<uchar> &vector, int64_t idx, int64_t val) -> void {
            ((int32_t *)vector.data())[idx] = val;
        };
    } else if (dataType == "INT64") {
        elementSize = sizeof(int64_t);
        setElement = [](std::vector<uchar> &vector, int64_t idx, int64_t val) -> void {
            ((int64_t *)vector.data())[idx] = val;
        };
    } else {
        std::cout << "Error: Unknown input data type: " << dataType << std::endl;
        exit(-1);
    }

    for (const auto &sequence : modelConfig["inputs"]) {
        availableVocabIds.emplace_back();
        availableAttentionMasks.emplace_back();
        const auto &vocabIds = sequence[0];
        const auto &attentionMasks = sequence[1];
        
        int seqLen = vocabIds.size();
        availableVocabIds.back().resize(seqLen * elementSize);
        availableAttentionMasks.back().resize(seqLen * elementSize);
        
        for (int i = 0; i < seqLen; ++i) {
            setElement(availableVocabIds.back(), i, vocabIds[i].asInt64());
            setElement(availableAttentionMasks.back(), i, attentionMasks[i].asInt64());
        }
    }
}

void VocabAttentionInputProcessor::prepare(int _batchSize)
{
    size_t maxSeqSize = 0;
    batchSize = _batchSize;
    selectedIdxs->clear();

    for (int i = 0; i < batchSize; ++i) {
        int idx = rand() % availableVocabIds.size();
        if (availableVocabIds[idx].size() > maxSeqSize) {
            maxSeqSize = availableVocabIds[idx].size();
        }
        selectedIdxs->push_back(idx);
    }

    maxSeqLength = maxSeqSize / elementSize;
}

std::shared_ptr<std::vector<int64_t>> VocabAttentionInputProcessor::preprocess(Json::Value &shapeVar)
{
    emptyBufferPool.clear();

    shapeVar["batch_size"] = batchSize;
    shapeVar["sequence"] = (Json::Value::Int64)maxSeqLength;

    tc::InferInput *vocabIdsTensor = tensors[0];
    tc::InferInput *attentionMasksTensor = tensors[1];

    vocabIdsTensor->SetShape({ batchSize, (int64_t)maxSeqLength });
    attentionMasksTensor->SetShape({ batchSize, (int64_t)maxSeqLength });
    
    vocabIdsTensor->Reset();
    attentionMasksTensor->Reset();

    for (int i = 0; i < batchSize; ++i) {
        ASSERT_TRITON_ERROR(vocabIdsTensor->AppendRaw(availableVocabIds[(*selectedIdxs)[i]]));
        ASSERT_TRITON_ERROR(attentionMasksTensor->AppendRaw(availableAttentionMasks[(*selectedIdxs)[i]]));
        
        if (availableVocabIds[(*selectedIdxs)[i]].size() < maxSeqLength * elementSize) {
            emptyBufferPool.emplace_back(maxSeqLength * elementSize - availableVocabIds[(*selectedIdxs)[i]].size(), 0);
            vocabIdsTensor->AppendRaw(emptyBufferPool.back());
            attentionMasksTensor->AppendRaw(emptyBufferPool.back());
        }
    }

    return selectedIdxs;
}