#include "accanalyzer.h"
#include "tfclient.h"

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
    TFClient *tfClient = dynamic_cast<TFClient *>(this->getClient());
    std::string inputLabel = tfClient->getInputLabel();
    std::string outputLabel = tfClient->getOutputLabel();

    totalInference += 1;
    correctInference += inputLabel == outputLabel;
}

Json::Value AccuarcyAnalyzer::generateReport()
{
    Json::Value report;
    report["type"] = "accuarcy";
    report["accuarcy"] = totalInference ? (float)correctInference / totalInference : 0;
    return report;
}