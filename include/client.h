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

class Client
{
private:
    std::string name;
    std::shared_ptr<BasicAnalyzer> basicAnalyzer;
    std::vector<std::shared_ptr<Analyzer>> customAnalyzers;

public:
    Client();
    void setName(const std::string &name);

    std::vector<std::shared_ptr<Record>> produceRecords();
    Json::Value generateReport();
    void addAnalyzer(std::shared_ptr<Analyzer> customAnalyzer);
    void initAnalyzers();
    void startAnalyzers(const std::chrono::system_clock::time_point &beginTime);
    void stopAnalyzers(const std::chrono::system_clock::time_point &endTime);

    virtual void init() {}
    virtual void prepareInput() {}
    virtual void preprocess() {}
    virtual void copyInput() {}
    virtual void infer() {}
    virtual void copyOutput() {}
    virtual void postprocess() {}
};

} // namespace DISB

#endif
