#include "analyzer.h"
#include "utils.h"

#include <list>
#include <chrono>
#include <memory>
#include <jsoncpp/json/json.h>

namespace DISB {

using std::chrono::system_clock;
using std::chrono::nanoseconds;
using std::chrono::microseconds;

Analyzer::Analyzer(): client(nullptr)
{

}

Client *Analyzer::getClient()
{
    return client;
}

void Analyzer::setClient(Client *clt)
{
    client = clt;
}

std::shared_ptr<Record> Analyzer::produceRecordWrapper()
{
    auto record = produceRecord();
    record->analyzer = this;
    return record;
}

void BasicAnalyzer::start(const system_clock::time_point &_beginTime)
{
    records.clear();
    beginTime = _beginTime;
}

void BasicAnalyzer::stop(const system_clock::time_point &_endTime)
{
    endTime = _endTime;
}

std::shared_ptr<Record> BasicAnalyzer::produceRecord()
{
    return std::make_shared<Record>();
}

void BasicAnalyzer::consumeRecord(std::shared_ptr<Record> record)
{
    records.push_back(record);
}

Json::Value BasicAnalyzer::generateReport()
{
    // req/s
    double avgThroughput(0);

    nanoseconds prepareInputLatencyAvg(0);
    nanoseconds preprocessLatencyAvg(0);
    nanoseconds copyInputLatencyAvg(0);
    nanoseconds inferLatencyAvg(0);
    nanoseconds copyOutputLatencyAvg(0);
    nanoseconds postprocessLatencyAvg(0);

    if (records.size() > 0) {
        // analysis records and calculate latency & throughput
        nanoseconds prepareInputLatencySum(0);
        nanoseconds preprocessLatencySum(0);
        nanoseconds copyInputLatencySum(0);
        nanoseconds inferLatencySum(0);
        nanoseconds copyOutputLatencySum(0);
        nanoseconds postprocessLatencySum(0);

        for (auto record : records) {
            std::shared_ptr<TimePoints> tps = record->timePoints;
            
            prepareInputLatencySum += tps->prepareInputEnd - tps->prepareInputBegin;
            preprocessLatencySum += tps->preprocessEnd - tps->preprocessBegin;
            copyInputLatencySum += tps->copyInputEnd - tps->copyInputBegin;
            inferLatencySum += tps->inferEnd - tps->inferBegin;
            copyOutputLatencySum += tps->copyOutputEnd - tps->copyOutputBegin;
            postprocessLatencySum += tps->postprocessEnd - tps->postprocessBegin;
        }

        prepareInputLatencyAvg = prepareInputLatencySum / records.size();
        preprocessLatencyAvg = preprocessLatencySum / records.size();
        copyInputLatencyAvg = copyInputLatencySum / records.size();
        inferLatencyAvg = inferLatencySum / records.size();
        copyOutputLatencyAvg = copyOutputLatencySum / records.size();
        postprocessLatencyAvg = postprocessLatencySum / records.size();

        nanoseconds benchmarkTime(endTime - beginTime);
        if (benchmarkTime.count() > 0) {
            avgThroughput = double(BILLION * records.size()) / benchmarkTime.count();
        }
    }
    
    Json::Value report;
    report["type"] = "basic";
    report["avgThroughput(req/s)"] = avgThroughput;

    // measured in microseconds
    report["avgPrepareInputLatency(us)"] = (Json::Value::Int64) prepareInputLatencyAvg.count() / THOUSAND;
    report["avgPreprocessLatency(us)"] = (Json::Value::Int64) preprocessLatencyAvg.count() / THOUSAND;
    report["avgCopyInputLatency(us)"] = (Json::Value::Int64) copyInputLatencyAvg.count() / THOUSAND;
    report["avgInferLatency(us)"] = (Json::Value::Int64) inferLatencyAvg.count() / THOUSAND;
    report["avgCopyOutputLatency(us)"] = (Json::Value::Int64) copyOutputLatencyAvg.count() / THOUSAND;
    report["avgPostprocessLatency(us)"] = (Json::Value::Int64) postprocessLatencyAvg.count() / THOUSAND;

    return report;
}

} // namespace DISB