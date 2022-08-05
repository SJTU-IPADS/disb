#ifndef _DISB_BENCHMARK_H_
#define _DISB_BENCHMARK_H_

#include "analyzer.h"
#include "strategy.h"
#include "client.h"

#include <vector>
#include <memory>
#include <string>
#include <jsoncpp/json/json.h>

#define LATENCY_MEASURE_TIME 5
#define DELAY_BEGIN_SECONDS 1

namespace DISB
{

struct BenchmarkTask
{
    std::shared_ptr<Client> client;
    std::shared_ptr<Strategy> strategy;
    std::shared_ptr<StandAloneLatency> standAloneLatency;

    BenchmarkTask(std::shared_ptr<Client> clt, std::shared_ptr<Strategy> stg):
        client(clt), strategy(stg), standAloneLatency(std::make_shared<StandAloneLatency>()) {}
    
    void runBenchmark(const std::chrono::system_clock::time_point &beginTime,
                      const std::chrono::system_clock::time_point &endTime);
};

class BenchmarkSuite
{
private:
    std::chrono::seconds benchmarkTime;
    std::chrono::system_clock::time_point beginTime;
    std::chrono::system_clock::time_point endTime;
    std::vector<BenchmarkTask> tasks;

public:
    BenchmarkSuite();
    ~BenchmarkSuite();

    void init(const std::string &configJsonStr,
              std::shared_ptr<Client> clientFactory(const Json::Value &config),
              std::shared_ptr<Strategy> strategyFactory(const Json::Value &config) = builtinStrategyFactory);

    void run(void strategyCoordinator(const std::vector<StrategyInfo> &strategyInfos) = builtinStrategyCoordinator);
    Json::Value generateReport();
};

} // namespace DISB

#endif
