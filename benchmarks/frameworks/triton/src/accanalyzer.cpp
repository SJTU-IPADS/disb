#include "accanalyzer.h"
#include "tritonclient.h"

AccuarcyAnalyzer::AccuarcyAnalyzer():
    totalInference(0), correctInference(0)
{

}

void AccuarcyAnalyzer::start(const std::chrono::system_clock::time_point &beginTime)
{
    totalInference = 0;
    correctInference = 0;
}
    
std::shared_ptr<DISB::Record> AccuarcyAnalyzer::produceRecord()
{
    return std::make_shared<DISB::Record>();
}

void AccuarcyAnalyzer::consumeRecord(std::shared_ptr<DISB::Record> record)
{
    TritonClient *tritonClient = dynamic_cast<TritonClient *>(this->getClient());
    totalInference += tritonClient->getBatchSize();
    correctInference += tritonClient->getCorrectInferCount();
}

Json::Value AccuarcyAnalyzer::generateReport()
{
    Json::Value report;
    report["type"] = "accuarcy";
    report["accuarcy"] = totalInference ? (float)correctInference / totalInference : 0;
    return report;
}