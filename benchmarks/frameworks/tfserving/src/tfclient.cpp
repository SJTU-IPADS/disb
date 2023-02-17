#include "tfclient.h"
#include "accanalyzer.h"

#include <fstream>
#include <iostream>

using tensorflow::TensorProto;
using tensorflow::TensorShapeProto;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

TFClient::TFClient(const Json::Value &config)
{
    this->setName(config["name"].asString());
    batchSize = config["batch_size"].asInt();
    modelName = config["model_name"].asString();
    benchmarksDir = config["benchmarks_dir"].asString();
    modelDir = joinPath(joinPath(benchmarksDir, "models"), modelName);
    
    std::string grpcAddr = config["tfs_grpc_addr"].asString();
    std::shared_ptr<grpc::Channel> channel =
        grpc::CreateChannel(grpcAddr, grpc::InsecureChannelCredentials());
    stub = PredictionService::NewStub(channel);
}

void TFClient::init()
{
    modelConfig = readJsonFromFile(joinPath(modelDir, "model.json"));

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

void TFClient::prepareInput()
{
    context = std::make_shared<grpc::ClientContext>();
    request = std::make_shared<PredictRequest>();
    response = std::make_shared<PredictResponse>();
    inputProcessor->prepare(batchSize);
    request->mutable_model_spec()->set_name(modelName);
}

void TFClient::preprocess()
{
    google::protobuf::Map<std::string, tensorflow::TensorProto> &inputs =
        *request->mutable_inputs();
    selectedIdxs = inputProcessor->preprocess(inputs, shapeVar);
}

void TFClient::infer()
{
    requestStatus = stub->Predict(context.get(), *request, response.get());
}

void TFClient::postprocess()
{
    bool inferSuccess = true;
    if (!requestStatus.ok()) {
        inferSuccess = false;
        std::cout << requestStatus.error_message() << "\n";
    }

    google::protobuf::Map<std::string, tensorflow::TensorProto> &outputs =
        *response->mutable_outputs();

    outputProcessor->processOutput(batchSize, inferSuccess, *selectedIdxs, outputs);
}

int TFClient::getBatchSize() const
{
    return batchSize;
}

int TFClient::getCorrectInferCount()
{
    return outputProcessor->checkCorrect();
}

std::shared_ptr<DISB::Client> tfClientFactory(const Json::Value &config)
{
    auto client = std::make_shared<TFClient>(config);
    client->addAnalyzer(std::make_shared<AccuarcyAnalyzer>());
    return client;
}
