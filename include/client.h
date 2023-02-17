#ifndef _DISB_CLIENT_H_
#define _DISB_CLIENT_H_

#include "analyzer.h"

#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <jsoncpp/json/json.h>

namespace DISB
{

class InferResult
{
public:
    int64_t resultSerialNumber = 0;
    std::string producerTaskId;
    std::string consumerTaskId;
};

class Client
{
private:
    std::string name;
    std::shared_ptr<BasicAnalyzer> basicAnalyzer;
    std::vector<std::shared_ptr<Analyzer>> customAnalyzers;

public:
    Client();
    std::string getName();
    void setName(const std::string &name);

    std::vector<std::shared_ptr<Record>> produceRecords();
    Json::Value generateReport();
    void addAnalyzer(std::shared_ptr<Analyzer> customAnalyzer);
    void initAnalyzers();
    void startAnalyzers(const std::chrono::system_clock::time_point &beginTime);
    void stopAnalyzers(const std::chrono::system_clock::time_point &endTime);
    void setStandAloneLatency(std::chrono::nanoseconds standAloneLatency);

    virtual void init() {}
    virtual void prepareInput() {}
    virtual void preprocess() {}
    virtual void copyInput() {}
    virtual void infer() {}
    virtual void copyOutput() {}
    virtual void postprocess() {}
    virtual std::shared_ptr<InferResult> produceResult() { return std::make_shared<InferResult>(); };
};

class DependentClient: public Client
{
public:
    virtual void consumePrevResults(const std::map<std::string, std::shared_ptr<InferResult>> &prevResults) {}
    virtual std::map<std::string, std::shared_ptr<InferResult>> produceDummyPrevResults()
    {
        return std::map<std::string, std::shared_ptr<InferResult>>();
    }
};

} // namespace DISB

#endif
