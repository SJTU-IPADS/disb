#pragma once

#include <thread>
#include "disb.h"
#include "reef/client/client.h"

using namespace reef;

#ifndef REEF_RESOURCE_DIR 
#define REEF_RESOURCE_DIR "./reef/resources/"
#endif

class REEFDISBClient: public DISB::Client
{
public:
    REEFDISBClient(const Json::Value &config);
    
    ~REEFDISBClient();
    
    virtual void init() override;

    virtual void copyInput() override;

    virtual void infer() override;

    virtual void copyOutput() override;
private:
    std::unique_ptr<client::REEFClient> client;
    std::shared_ptr<client::ModelHandle> model;

    std::string model_name;
    int priority;
};

std::shared_ptr<DISB::Client> reef_client_factory(const Json::Value &config);
