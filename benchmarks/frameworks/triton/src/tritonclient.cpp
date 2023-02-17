#include "tritonclient.h"
#include "triton_utils.h"
#include "accanalyzer.h"

namespace tc = triton::client;

TritonClient::TritonClient(const Json::Value &config)
{
    this->setName(config["name"].asString());
    batchSize = config["batch_size"].asInt();
    modelName = config["model_name"].asString();
    benchmarksDir = config["benchmarks_dir"].asString();
    modelDir = joinPath(joinPath(benchmarksDir, "models"), modelName);

    std::string grpcAddr = config["triton_grpc_addr"].asString();
    inferOptions = std::make_unique<tc::InferOptions>(modelName);
    ASSERT_TRITON_ERROR(tc::InferenceServerGrpcClient::Create(&grpcClient, grpcAddr));
}

void TritonClient::init()
{
    inference::ModelConfigResponse tcModelConfig;
    inference::ModelMetadataResponse tcModelMetadata;
    ASSERT_TRITON_ERROR(grpcClient->ModelConfig(&tcModelConfig, modelName));
    ASSERT_TRITON_ERROR(grpcClient->ModelMetadata(&tcModelMetadata, modelName));

    assert(tcModelMetadata.versions_size() > 0);
    inferOptions->model_version_ = tcModelMetadata.versions(0);

    modelConfig = readJsonFromFile(joinPath(modelDir, "model.json"));

    std::string inputConvention = modelConfig["input_convention"].asString();
    if (inputConvention == "common_image_input") {
        inputProcessor = std::make_shared<CommonImageInputProcessor>(modelDir, modelConfig, tcModelMetadata);
    } else if (inputConvention == "vocab_attention") {
        inputProcessor = std::make_shared<VocabAttentionInputProcessor>(modelConfig, tcModelMetadata);
    } else {
        std::cout << "Error: Unknown input convention: " << inputConvention << std::endl;
        exit(-1);
    }

    std::string outputConvention = modelConfig["output_convention"].asString();
    if (outputConvention == "softmax") {
        outputProcessor = std::make_shared<SoftmaxOutputProcessor>(modelConfig, tcModelMetadata);
    } else if (outputConvention == "hidden_state") {
        outputProcessor = std::make_shared<HiddenStateProcessor>(modelConfig, tcModelMetadata);
    } else {
        std::cout << "Error: Unknown output convention: " << outputConvention << std::endl;
        exit(-1);
    }
}

void TritonClient::prepareInput()
{
    inputProcessor->prepare(batchSize);
}

void TritonClient::preprocess()
{
    selectedIdxs = inputProcessor->preprocess(shapeVar);
}

void TritonClient::infer()
{
    tc::InferResult *result;
    ASSERT_TRITON_ERROR(grpcClient->Infer(&result, *inferOptions, inputProcessor->inputTensors(), outputProcessor->outputTensors()));
    inferResult = std::shared_ptr<tc::InferResult>(result);
}

void TritonClient::postprocess()
{
    bool inferSuccess = true;
    if (!inferResult->RequestStatus().IsOk()) {
        inferSuccess = false;
        std::cout << "Error: triton client " << this->getName() << " infer failed: " << inferResult->RequestStatus().Message() << "\n";
    }

    outputProcessor->processOutput(batchSize, inferSuccess, *selectedIdxs, inferResult);
}

int TritonClient::getBatchSize() const
{
    return batchSize;
}

int TritonClient::getCorrectInferCount()
{
    return outputProcessor->checkCorrect();
}

std::shared_ptr<DISB::Client> tritonClientFactory(const Json::Value &config)
{
    auto client = std::make_shared<TritonClient>(config);
    client->addAnalyzer(std::make_shared<AccuarcyAnalyzer>());
    return client;
}