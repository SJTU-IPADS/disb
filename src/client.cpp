#include "client.h"

#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <jsoncpp/json/json.h>

namespace DISB
{

using std::chrono::system_clock;

Client::Client():
    name("Unknown"),
    basicAnalyzer(std::make_shared<BasicAnalyzer>())
{

}

void Client::setName(const std::string &_name)
{
    name = _name;
}

std::vector<std::shared_ptr<Record>> Client::produceRecords()
{
    std::shared_ptr<TimePoints> timePoints = std::make_shared<TimePoints>();
    std::vector<std::shared_ptr<Record>> records;

    auto record = basicAnalyzer->produceRecordWrapper();
    record->timePoints = timePoints;
    records.push_back(record);

    for (auto customAnalyzer : customAnalyzers) {
        auto customRecord = customAnalyzer->produceRecordWrapper();
        customRecord->timePoints = timePoints;
        records.push_back(customRecord);
    }

    return records;
}

Json::Value Client::generateReport()
{
    Json::Value analyzerReports;
    
    analyzerReports.append(basicAnalyzer->generateReport());
    for (auto customAnalyzer : customAnalyzers) {
        analyzerReports.append(customAnalyzer->generateReport());
    }

    Json::Value report;
    report["clientName"] = name;
    report["analyzers"] = analyzerReports;
    return report;
}

void Client::addAnalyzer(std::shared_ptr<Analyzer> customAnalyzer)
{
    customAnalyzer->setClient(this);
    customAnalyzers.push_back(customAnalyzer);
}

void Client::initAnalyzers()
{
    basicAnalyzer->init();
    for (auto customAnalyzer : customAnalyzers) {
        customAnalyzer->init();
    }
}

void Client::startAnalyzers(const std::chrono::system_clock::time_point &beginTime)
{
    basicAnalyzer->start(beginTime);
    for (auto customAnalyzer : customAnalyzers) {
        customAnalyzer->start(beginTime);
    }
}

void Client::stopAnalyzers(const std::chrono::system_clock::time_point &endTime)
{
    basicAnalyzer->stop(endTime);
    for (auto customAnalyzer : customAnalyzers) {
        customAnalyzer->stop(endTime);
    }
}

} // namespace DISB
