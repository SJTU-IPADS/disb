#include "client.h"
#include "analyzer.h"
#include "disb_utils.h"

#include <list>
#include <chrono>
#include <memory>
#include <fstream>
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

void BasicAnalyzer::setStandAloneLatency(std::chrono::nanoseconds _standAloneLatency)
{
    standAloneLatency = _standAloneLatency;
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
    nanoseconds totalLatencyAvg(0);

    if (records.size() > 0) {
        // analysis records and calculate latency & throughput
        nanoseconds prepareInputLatencySum(0);
        nanoseconds preprocessLatencySum(0);
        nanoseconds copyInputLatencySum(0);
        nanoseconds inferLatencySum(0);
        nanoseconds copyOutputLatencySum(0);
        nanoseconds postprocessLatencySum(0);

        // std::string name = records.front()->analyzer->getClient()->getName();
        // std::ofstream f(std::string("./") + name + ".txt");
        // int inferId = 0;

        for (auto record : records) {
            std::shared_ptr<TimePoints> tps = record->timePoints;

            // f << "infer id: " << inferId++ << "\n";
            // f << "prepare: " << (tps->prepareInputBegin - beginTime).count()/1000 << "us ~ " << (tps->prepareInputEnd - beginTime).count()/1000 << "us\n";
            // f << "preprocess: " << (tps->preprocessBegin - beginTime).count()/1000 << "us ~ " << (tps->preprocessEnd - beginTime).count()/1000 << "us\n";
            // f << "copy: " << (tps->copyInputBegin - beginTime).count()/1000 << "us ~ " << (tps->copyInputEnd - beginTime).count()/1000 << "us\n";
            // f << "infer: " << (tps->inferBegin - beginTime).count()/1000 << "us ~ " << (tps->inferEnd - beginTime).count()/1000 << "us\n";
            // f << "copy: " << (tps->copyOutputBegin - beginTime).count()/1000 << "us ~ " << (tps->copyOutputEnd - beginTime).count()/1000 << "us\n";
            // f << "postprocess: " << (tps->postprocessBegin - beginTime).count()/1000 << "us ~ " << (tps->postprocessEnd - beginTime).count()/1000 << "us\n\n\n";
            
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
        totalLatencyAvg = prepareInputLatencyAvg
                        + preprocessLatencyAvg
                        + copyInputLatencyAvg
                        + inferLatencyAvg
                        + copyOutputLatencyAvg
                        + postprocessLatencyAvg;

        nanoseconds benchmarkTime(endTime - beginTime);
        if (benchmarkTime.count() > 0) {
            avgThroughput = double(BILLION * records.size()) / benchmarkTime.count();
        }
    }
    
    Json::Value report;
    report["type"] = "basic";
    report["avgThroughput(req/s)"] = avgThroughput;

    // measured in microseconds
    report["standAloneTotalLatency(us)"] = (Json::Value::Int64) standAloneLatency.count() / THOUSAND;
    report["avgPrepareInputLatency(us)"] = (Json::Value::Int64) prepareInputLatencyAvg.count() / THOUSAND;
    report["avgPreprocessLatency(us)"] = (Json::Value::Int64) preprocessLatencyAvg.count() / THOUSAND;
    report["avgCopyInputLatency(us)"] = (Json::Value::Int64) copyInputLatencyAvg.count() / THOUSAND;
    report["avgInferLatency(us)"] = (Json::Value::Int64) inferLatencyAvg.count() / THOUSAND;
    report["avgCopyOutputLatency(us)"] = (Json::Value::Int64) copyOutputLatencyAvg.count() / THOUSAND;
    report["avgPostprocessLatency(us)"] = (Json::Value::Int64) postprocessLatencyAvg.count() / THOUSAND;
    report["avgTotalLatency(us)"] = (Json::Value::Int64) totalLatencyAvg.count() / THOUSAND;
    report["avgTotalLatencyIncrease(us)"] = (Json::Value::Int64) (totalLatencyAvg.count() - standAloneLatency.count()) / THOUSAND;

    return report;
}

} // namespace DISB