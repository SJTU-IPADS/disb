#ifndef _DISB_BENCHMARK_H_
#define _DISB_BENCHMARK_H_

#include "analyzer.h"
#include "load.h"
#include "client.h"

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <jsoncpp/json/json.h>

#define WARMUP_TIME                         100
#define STANDALONE_LATENCY_MEASURE_TIME     200
#define DELAY_BEGIN_SECONDS                 5

namespace DISB
{

struct BenchmarkTask
{
    bool isDependent = false;
    std::string id;
    std::vector<BenchmarkTask *> nextTasks;
    std::shared_ptr<Client> client;
    std::shared_ptr<Load> load;
    std::shared_ptr<StandAloneLatency> standAloneLatency;

    BenchmarkTask(): id("UNKNOWN") {}
    BenchmarkTask(std::string taskId, std::shared_ptr<Client> clt, std::shared_ptr<Load> stg);
    
    void inferOnce();
    void runBenchmark(const std::chrono::system_clock::time_point &beginTime,
                      const std::chrono::system_clock::time_point &endTime);
};

class BenchmarkSuite
{
private:
    std::chrono::seconds benchmarkTime;
    std::chrono::system_clock::time_point beginTime;
    std::chrono::system_clock::time_point endTime;
    std::map<std::string, BenchmarkTask> tasks;

public:
    BenchmarkSuite();
    ~BenchmarkSuite();

    void init(const std::string &configJsonStr,
              std::shared_ptr<Client> clientFactory(const Json::Value &config),
              std::shared_ptr<Load> loadFactory(const Json::Value &config) = builtinLoadFactory);
    
    void init(const Json::Value &configJson,
              std::shared_ptr<Client> clientFactory(const Json::Value &config),
              std::shared_ptr<Load> loadFactory(const Json::Value &config) = builtinLoadFactory);

    void run(void loadCoordinator(const std::vector<LoadInfo> &loadInfos) = builtinLoadCoordinator);
    Json::Value generateReport();
};

} // namespace DISB

#endif
