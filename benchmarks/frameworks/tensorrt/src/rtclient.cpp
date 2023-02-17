#include "utils.h"
#include "tensor.h"
#include "rtclient.h"
#include "accanalyzer.h"

#include <cmath>
#include <memory>
#include <chrono>
#include <iostream>

std::mutex TensorRTClient::enginePoolMtx;
std::map<std::string, std::shared_ptr<TensorRTEngine>> TensorRTClient::enginePool;

TensorRTClient::TensorRTClient(const Json::Value &config)
{
    this->setName(config["name"].asString());
    batchSize = config["batch_size"].asInt();
    modelName = config["model_name"].asString();
    benchmarksDir = config["benchmarks_dir"].asString();
    memoryLimit = config["memory_limit"].asInt64();
    modelDir = joinPath(joinPath(benchmarksDir, "models"), modelName);
    profileIdx = config["profile_idx"].asInt();
    profileCnt = config["profile_cnt"].asInt();

    if (cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking) != cudaSuccess) {
        printf("error: fail to create cuda stream\n");
    }
}

TensorRTClient::~TensorRTClient()
{
    cudaStreamDestroy(cudaStream);
}

void TensorRTClient::init()
{
    modelConfig = readJsonFromFile(joinPath(modelDir, "model.json"));

    std::shared_ptr<TensorRTEngine> trtEngine;
    std::unique_lock<std::mutex> lock(enginePoolMtx);
    auto it = enginePool.find(modelName);
    if (it == enginePool.end()) {
        trtEngine = std::make_shared<TensorRTEngine>(modelConfig, modelDir, profileCnt, memoryLimit, cudaStream);
        enginePool[modelName] = trtEngine;
    } else {
        trtEngine = it->second;
    }

    shapeVar = trtEngine->shapeVarOpt;
    maxBatchSize = trtEngine->maxBatchSize;
    enableExplicitBatchSize = trtEngine->enableExplicitBatchSize;

    const Json::Value &inputTensorShapes = modelConfig["input_tensor_shapes"];
    const Json::Value &inputTensorNames = modelConfig["input_tensor_names_onnx"];
    for (int i = 0; i < inputTensorNames.size(); ++i) {
        inputTensors.emplace_back(std::make_shared<Tensor>(inputTensorNames[i].asString(), inputTensorShapes[i]));
    }

    const Json::Value &outputTensorShapes = modelConfig["output_tensor_shapes"];
    const Json::Value &outputTensorNames = modelConfig["output_tensor_names_onnx"];
    for (int i = 0; i < outputTensorNames.size(); ++i) {
        outputTensors.emplace_back(std::make_shared<Tensor>(outputTensorNames[i].asString(), outputTensorShapes[i]));
    }

    context = std::unique_ptr<nvinfer1::IExecutionContext>(trtEngine->engine->createExecutionContext());
    tensorBuffer = std::make_unique<TensorBuffer>(trtEngine->engine, maxBatchSize, profileIdx, trtEngine->shapeVarMax, inputTensors, outputTensors);

    context->setOptimizationProfileAsync(profileIdx, cudaStream);
    cudaStreamSynchronize(cudaStream);

    std::string inputConvention = modelConfig["input_convention"].asString();
    if (inputConvention == "common_image_input") {
        inputProcessor = std::make_shared<CommonImageInputProcessor>(modelDir, modelConfig);
    } else if (inputConvention == "vocab_attention") {
        inputProcessor = std::make_shared<VocabAttentionInputProcessor>(modelConfig);
    } else {
        std::cout << "Error: Unknown input convention: " << inputConvention << std::endl;
        exit(-1);
    }

    std::string outputConvention = modelConfig["output_convention"].asString();
    if (outputConvention == "softmax") {
        outputProcessor = std::make_shared<SoftmaxOutputProcessor>(modelConfig);
    } else if (outputConvention == "hidden_state") {
        outputProcessor = std::make_shared<HiddenStateProcessor>(modelConfig);
    } else {
        std::cout << "Error: Unknown output convention: " << outputConvention << std::endl;
        exit(-1);
    }
}

void TensorRTClient::prepareInput()
{
    inputProcessor->prepare(batchSize);
}

void TensorRTClient::preprocess()
{
    selectedIdxs = inputProcessor->preprocess(inputTensors, shapeVar);
}

void TensorRTClient::copyInput()
{
    // Memcpy from host input buffers to device input buffers
    for (const auto &inputTensor : inputTensors) {
        inputTensor->setDims(shapeVar);
        inputTensor->copyToDeviceAsync(cudaStream);
    }
    cudaStreamSynchronize(cudaStream);
}

void TensorRTClient::infer()
{
    if (enableExplicitBatchSize) {
        for (const auto &inputTensor : inputTensors) {
            context->setBindingDimensions(inputTensor->binding, inputTensor->dims);
        }
    }
    inferSuccess = context->enqueueV2(tensorBuffer->getDeviceBuffers(), cudaStream, nullptr);
    cudaStreamSynchronize(cudaStream);
}

void TensorRTClient::copyOutput()
{
    // Memcpy from device output buffers to host output buffers
    for (const auto &outputTensor : outputTensors) {
        outputTensor->setDims(shapeVar);
        outputTensor->copyToHostAsync(cudaStream);
    }
    cudaStreamSynchronize(cudaStream);
}

void TensorRTClient::postprocess()
{
    outputProcessor->processOutput(inferSuccess, outputTensors, *selectedIdxs);
}

int TensorRTClient::getBatchSize() const
{
    return batchSize;
}

int TensorRTClient::getCorrectInferCount()
{
    return outputProcessor->checkCorrect();
}

std::shared_ptr<DISB::Client> rtClientFactory(const Json::Value &config)
{
    auto client = std::make_shared<TensorRTClient>(config);
    client->addAnalyzer(std::make_shared<AccuarcyAnalyzer>());
    return client;
}
