#ifndef _DISB_TF_TFSCLIENT_H_
#define _DISB_TF_TFSCLIENT_H_

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <google/protobuf/map.h>

#include "tensorflow/core/framework/types.grpc.pb.h"
#include "tensorflow/core/framework/tensor.grpc.pb.h"
#include "tensorflow/core/framework/tensor_shape.grpc.pb.h"
#include "tensorflow_serving/apis/predict.grpc.pb.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "disb.h"
#include "input.h"
#include "output.h"

class TFClient: public DISB::Client
{
public:
    TFClient(const Json::Value &config);

    virtual void init() override;
    virtual void prepareInput() override;
    virtual void preprocess() override;
    virtual void infer() override;
    virtual void postprocess() override;

    int getBatchSize() const;
    int getCorrectInferCount();
    
private:
    std::string modelName;
    std::string modelDir;
    std::string benchmarksDir;
    Json::Value modelConfig;
    
    int batchSize;
    Json::Value shapeVar;
    std::shared_ptr<InputProcessor> inputProcessor;
    std::shared_ptr<OutputProcessor> outputProcessor;
    std::shared_ptr<std::vector<int64_t>> selectedIdxs;

    grpc::Status requestStatus;
    std::shared_ptr<grpc::ClientContext> context;
    std::shared_ptr<tensorflow::serving::PredictRequest> request;
    std::shared_ptr<tensorflow::serving::PredictResponse> response;
    std::unique_ptr<tensorflow::serving::PredictionService::Stub> stub;
};

std::shared_ptr<DISB::Client> tfClientFactory(const Json::Value &config);

#endif