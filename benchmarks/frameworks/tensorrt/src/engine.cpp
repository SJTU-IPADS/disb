#include "engine.h"
#include "tensor.h"
#include "disb_utils.h"

#include <NvOnnxParser.h>

#define DLACORE -1

TensorRTEngine::TensorRTEngine(const Json::Value &modelConfig,
                               const std::string &modelDir,
                               int profileCnt, int64_t memoryLimit, cudaStream_t cudaStream)
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto profile = builder->createOptimizationProfile();

    shapeVarMin = modelConfig["tensor_shape_variables"]["min"];
    shapeVarOpt = modelConfig["tensor_shape_variables"]["opt"];
    shapeVarMax = modelConfig["tensor_shape_variables"]["max"];

    const Json::Value &inputTensorShapes = modelConfig["input_tensor_shapes"];
    const Json::Value &inputTensorNames = modelConfig["input_tensor_names_onnx"];
    
    for (int i = 0; i < inputTensorNames.size(); ++i) {
        nvinfer1::Dims inputDimsMin = convertDims(inputTensorShapes[i], shapeVarMin);
        nvinfer1::Dims inputDimsOpt = convertDims(inputTensorShapes[i], shapeVarOpt);
        nvinfer1::Dims inputDimsMax = convertDims(inputTensorShapes[i], shapeVarMax);

        const std::string inputTensorName = inputTensorNames[i].asString();

        profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMIN, inputDimsMin);
        profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kOPT, inputDimsOpt);
        profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMAX, inputDimsMax);
    }

    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    for (int i = 0; i < profileCnt; ++i) builder_config->addOptimizationProfile(profile);
    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, memoryLimit << 20);
    std::cout << "Limit memory pool to " << memoryLimit << " MiB\n";
    enableDLA(builder.get(), builder_config.get(), DLACORE);
    builder_config->setProfileStream(cudaStream);

    uint32_t explicitBatch = 0U;
    enableExplicitBatchSize = false;
    if (!shapeVarMax["batch_size"].isNull()) {
        enableExplicitBatchSize = true;
        maxBatchSize = shapeVarMax["batch_size"].asInt();
        explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    }
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    std::string modelPath = joinPath(modelDir, modelConfig["onnx_model"].asString());
    bool parsed = parser->parseFromFile(modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << "Error: " << parser->getError(i)->desc() << std::endl;
        }
    }

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *builder_config));
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
}