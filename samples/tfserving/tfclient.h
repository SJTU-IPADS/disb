#ifndef _DISB_TF_TFSCLIENT_H_
#define _DISB_TF_TFSCLIENT_H_

#include "disb.h"
#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <google/protobuf/map.h>

#include "rpc/tensorflow/core/framework/types.grpc.pb.h"
#include "rpc/tensorflow/core/framework/tensor.grpc.pb.h"
#include "rpc/tensorflow/core/framework/tensor_shape.grpc.pb.h"
#include "rpc/tensorflow_serving/apis/predict.grpc.pb.h"
#include "rpc/tensorflow_serving/apis/prediction_service.grpc.pb.h"

class TFClient: public DISB::Client
{
public:
    TFClient(const Json::Value &config);

    virtual void init() override;
    virtual void prepareInput() override;
    virtual void copyInput() override;
    virtual void infer() override;
    virtual void postprocess() override;

    std::string getInputLabel();
    std::string getOutputLabel();
    
private:
    std::string assetsDir;
    std::string modelName;
    std::vector<std::string> labels;
    std::vector<std::string> availableLabels;

    cv::Mat inputImage;
    std::string inputLabel;
    std::string outputLabel;

    grpc::Status requestStatus;
    std::shared_ptr<grpc::ClientContext> context;
    std::shared_ptr<tensorflow::serving::PredictRequest> request;
    std::shared_ptr<tensorflow::serving::PredictResponse> response;
    std::unique_ptr<tensorflow::serving::PredictionService::Stub> stub;
};

std::shared_ptr<DISB::Client> tfClientFactory(const Json::Value &config);

#endif