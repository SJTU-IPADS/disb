#include "utils.h"
#include "buffer.h"
#include "rtclient.h"
#include "accanalyzer.h"

#include <cmath>
#include <memory>
#include <chrono>
#include <iostream>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

#define DLACORE -1

using namespace nvonnxparser;

TensorRTClient::TensorRTClient(const Json::Value &config)
{
    modelName = config["name"].asString();
    modelPath = config["modelPath"].asString();
    inputDataPath = config["inputDataPath"].asString();

    setName(modelName);

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));

    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    
    auto builder_config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    auto parser = std::unique_ptr<IParser>(createParser(*network, logger));

    bool parsed = parser->parseFromFile(modelPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    if (!parsed) {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << "error" << parser->getError(i)->desc() << std::endl;
        }
    }

    builder_config->setMaxWorkspaceSize(16_MiB);
    builder_config->setFlag(BuilderFlag::kFP16);

    enableDLA(builder.get(), builder_config.get(), DLACORE);

    auto profileStream = makeCudaStream();
    builder_config->setProfileStream(*profileStream);

    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *builder_config));
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));

    mEngine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    
    auto mInputDims = network->getInput(0)->getDimensions();
    auto mOutputDims = network->getOutput(0)->getDimensions();

    inputH = mInputDims.d[2];
    inputW = mInputDims.d[3];
    outputSize = mOutputDims.d[1];

    srand(unsigned(time(nullptr)));
}

void TensorRTClient::init()
{
    buffers = std::make_unique<BufferManager>(mEngine);
    context = std::unique_ptr<IExecutionContext>(mEngine->createExecutionContext());
}

void TensorRTClient::prepareInput()
{
    inputNumber = rand() % 10;
    inputImage = cv::imread(inputDataPath + "/" + std::to_string(inputNumber) + ".png");
}

void TensorRTClient::preprocess()
{
    cv::cvtColor(inputImage, inputImage, cv::COLOR_RGB2GRAY);
    cv::resize(inputImage, inputImage, cv::Size(inputW, inputH));

    // Read the input data into the managed buffers
    float* hostDataBuffer = static_cast<float*>(buffers->getHostBuffer("Input3"));
    int pixelCnt = inputH * inputW;
    for (int i = 0; i < pixelCnt; ++i) {
        hostDataBuffer[i] = 1.0 -inputImage.data[i] / 255.0;
    }
}

void TensorRTClient::copyInput()
{
    // Memcpy from host input buffers to device input buffers
    buffers->copyInputToDevice();
}

void TensorRTClient::infer()
{
    bool status = context->executeV2(buffers->getDeviceBindings().data());
    if (!status) {
        return;
    }
}

void TensorRTClient::copyOutput()
{
    // Memcpy from device output buffers to host output buffers
    buffers->copyOutputToHost();
}

void TensorRTClient::postprocess()
{
    float* output = static_cast<float*>(buffers->getHostBuffer("Plus214_Output_0"));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++) {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    for (int i = 0; i < outputSize; i++) {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i]) {
            idx = i;
        }
    }

    outputNumber = idx;
}

int TensorRTClient::getInputNumber()
{
    return inputNumber;
}

int TensorRTClient::getOutputNumber()
{
    return outputNumber;
}

std::shared_ptr<DISB::Client> rtClientFactory(const Json::Value &config)
{
    auto client = std::make_shared<TensorRTClient>(config);
    client->addAnalyzer(std::make_shared<AccuarcyAnalyzer>());
    return client;
}
