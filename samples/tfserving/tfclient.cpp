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
    assetsDir = config["assets_dir"].asString();
    modelName = config["model_name"].asString();
    std::string grpcAddr = config["grpc_addr"].asString();

    std::shared_ptr<grpc::Channel> channel =
        grpc::CreateChannel(grpcAddr, grpc::InsecureChannelCredentials());
    stub = PredictionService::NewStub(channel);
}

void TFClient::init()
{
    std::ifstream labelsFile(assetsDir + "/labels.txt");
    std::string label;
    while (getline(labelsFile, label)) {
        labels.push_back(label);
    }

    availableLabels.push_back("cassette");
    availableLabels.push_back("gorilla");
    availableLabels.push_back("pizza");

    srand(unsigned(time(nullptr)));
}

void TFClient::prepareInput()
{
    inputLabel = availableLabels[rand() % availableLabels.size()];
    inputImage = cv::imread(assetsDir + "/inputs/" + inputLabel + ".png");
    cv::resize(inputImage, inputImage, cv::Size(224, 224));
}

void TFClient::copyInput()
{
    context = std::make_shared<grpc::ClientContext>();
    request = std::make_shared<PredictRequest>();
    response = std::make_shared<PredictResponse>();
    request->mutable_model_spec()->set_name(modelName);

    google::protobuf::Map<std::string, tensorflow::TensorProto>& inputs =
        *request->mutable_inputs();

    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_FLOAT);

    for (int row = 0; row < 224; ++row) {
        for (int col = 0; col < 224; ++col) {
            auto &rgb = inputImage.at<cv::Vec3b>(row, col);
            proto.add_float_val((255 - rgb[0]) / 255.0);
            proto.add_float_val((255 - rgb[1]) / 255.0);
            proto.add_float_val((255 - rgb[2]) / 255.0);
        }
    }

    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    proto.mutable_tensor_shape()->add_dim()->set_size(224);
    proto.mutable_tensor_shape()->add_dim()->set_size(224);
    proto.mutable_tensor_shape()->add_dim()->set_size(3);

    inputs["inputs"] = proto;
}

void TFClient::infer()
{
    requestStatus = stub->Predict(context.get(), *request, response.get());
}

void TFClient::postprocess()
{
    if (!requestStatus.ok()) {
        return;
    }

    google::protobuf::Map<std::string, tensorflow::TensorProto> &outputs =
        *response->mutable_outputs();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
        std::string section = it->first;
        if (section != "logits") {
            continue;
        }

        tensorflow::TensorProto &result_tensor_proto = it->second;
        int outputSize = result_tensor_proto.float_val_size();
        int outputIdx = 0;
        float maxLogit = std::numeric_limits<float>().min();
        for (size_t i = 0; i < outputSize; ++i) {
            float logit = result_tensor_proto.float_val(i);
            if (logit > maxLogit) {
                maxLogit = logit;
                outputIdx = i;
            }
        }
        outputLabel = labels[outputIdx];
    }
}

std::string TFClient::getInputLabel()
{
    return inputLabel;
}

std::string TFClient::getOutputLabel()
{
    return outputLabel;
}

std::shared_ptr<DISB::Client> tfClientFactory(const Json::Value &config)
{
    auto client = std::make_shared<TFClient>(config);
    client->addAnalyzer(std::make_shared<AccuarcyAnalyzer>());
    return client;
}
