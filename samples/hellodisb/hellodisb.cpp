#include "disb.h"

#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <iostream>
#include <jsoncpp/json/json.h>

class HelloClient: public DISB::Client
{
public:
    HelloClient(const std::string &name)
    {
        setName(name);
    }
    
    ~HelloClient() {}
    
    virtual void prepareInput() override
    {
        std::this_thread::sleep_for(std::chrono::microseconds(105));
    }

    virtual void preprocess() override
    {
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    }

    virtual void copyInput() override
    {
        std::this_thread::sleep_for(std::chrono::microseconds(35));
    }

    virtual void infer() override
    {
        std::this_thread::sleep_for(std::chrono::microseconds(535));
    }

    virtual void copyOutput() override
    {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    virtual void postprocess() override
    {
        std::this_thread::sleep_for(std::chrono::microseconds(15));
    }
};

std::shared_ptr<DISB::Client> helloClientFactory(const Json::Value &config)
{
    return std::make_shared<HelloClient>(config["name"].asString());
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: hellodisb config.json" << std::endl;
        return -1;
    }

    DISB::BenchmarkSuite benchmark;
    std::string jsonStr = readStringFromFile(argv[1]);
    benchmark.init(jsonStr, helloClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << report << std::endl;

    return 0;
}
