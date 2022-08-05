#include "benchmark.h"
#include "utils.h"

#include <vector>
#include <thread>
#include <memory>
#include <jsoncpp/json/json.h>

namespace DISB
{

using std::chrono::system_clock;
using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

void BenchmarkTask::runBenchmark(const system_clock::time_point &beginTime,
                                   const system_clock::time_point &endTime)
{
    strategy->start(beginTime);
    client->startAnalyzers(beginTime);

    system_clock::time_point now = system_clock::now();
    system_clock::time_point nextLaunchTime = strategy->nextLaunchTime(now);
    
    while (nextLaunchTime < endTime) {
        now = system_clock::now();
        if (now < nextLaunchTime) {
            std::this_thread::sleep_for(nextLaunchTime - now);
        }

        std::vector<std::shared_ptr<Record>> records = client->produceRecords();
        // records should always be not empty
        // because a basic analyzer will always be there, producing at least one record
        // for safety's sake, here will check records' length
        std::shared_ptr<TimePoints> timePoints = records.empty()
                                               ? std::make_shared<TimePoints>()
                                               : records.front()->timePoints;

        // trigger analyers' prepare input event and record execution time
        for (auto record : records) record->analyzer->onPrepareInput(record);
        timePoints->prepareInputBegin = system_clock::now();
        client->prepareInput();
        timePoints->prepareInputEnd = system_clock::now();

        // trigger analyers' preprocess event and record execution time
        for (auto record : records) record->analyzer->onPreprocess(record);
        timePoints->preprocessBegin = system_clock::now();
        client->preprocess();
        timePoints->preprocessEnd = system_clock::now();

        // trigger analyers' copy input event and record execution time
        for (auto record : records) record->analyzer->onCopyInput(record);
        timePoints->copyInputBegin = system_clock::now();
        client->copyInput();
        timePoints->copyInputEnd = system_clock::now();

        // trigger analyers' infer event and record execution time
        for (auto record : records) record->analyzer->onInfer(record);
        timePoints->inferBegin = system_clock::now();
        client->infer();
        timePoints->inferEnd = system_clock::now();

        // trigger analyers' copy output event and record execution time
        for (auto record : records) record->analyzer->onCopyOutput(record);
        timePoints->copyOutputBegin = system_clock::now();
        client->copyOutput();
        timePoints->copyOutputEnd = system_clock::now();

        // trigger analyers' postprocess event and record execution time
        for (auto record : records) record->analyzer->onPostprocess(record);
        timePoints->postprocessBegin = system_clock::now();
        client->postprocess();
        timePoints->postprocessEnd = system_clock::now();

        // give the record back to the analyzer produced it
        for (auto record : records) record->analyzer->consumeRecord(record);

        now = system_clock::now();
        nextLaunchTime = strategy->nextLaunchTime(now);
    }

    strategy->stop(endTime);
    client->stopAnalyzers(endTime);
}

BenchmarkSuite::BenchmarkSuite()
{

}

BenchmarkSuite::~BenchmarkSuite()
{

}

void BenchmarkSuite::init(const std::string &configJsonStr,
                          std::shared_ptr<Client> clientFactory(const Json::Value &config),
                          std::shared_ptr<Strategy> strategyFactory(const Json::Value &config))
{
    Json::Reader reader;
    Json::Value config;

    reader.parse(configJsonStr, config, false);

    benchmarkTime = seconds(config["time"].asInt());
    for (Json::Value taskConfig : config["tasks"]) {
        std::shared_ptr<Client> client = clientFactory(taskConfig["client"]);
        std::shared_ptr<Strategy> strategy = strategyFactory(taskConfig["strategy"]);

        // check if the factory method is correct
        if (client.get() != nullptr && dynamic_cast<Client *>(client.get()) != nullptr
            && strategy.get() != nullptr && dynamic_cast<Strategy *>(strategy.get()) != nullptr) {
            
            client->init();
            client->initAnalyzers();
            strategy->init();
            tasks.emplace_back(client, strategy);
        }
    }
}

void BenchmarkSuite::run(void strategyCoordinator(const std::vector<StrategyInfo> &strategyInfos))
{
    // 1. warm up
    for (auto &task : tasks) {
        task.client->prepareInput();
        task.client->preprocess();
        task.client->copyInput();
        task.client->infer();
        task.client->copyOutput();
        task.client->postprocess();
    }

    // 2. measure the stand alone latency of each client
    for (auto &task : tasks) {
        // sleep 100 ms to reduce influence between two tasks
        std::this_thread::sleep_for(microseconds(100000));

        nanoseconds prepareInputLatencySum(0);
        nanoseconds preprocessLatencySum(0);
        nanoseconds copyInputLatencySum(0);
        nanoseconds inferLatencySum(0);
        nanoseconds copyOutputLatencySum(0);
        nanoseconds postprocessLatencySum(0);

        for (int i = 0; i < LATENCY_MEASURE_TIME; ++i) {
            auto begin = system_clock::now();
            task.client->prepareInput();
            auto end = system_clock::now();
            prepareInputLatencySum += end - begin;

            begin = system_clock::now();
            task.client->preprocess();
            end = system_clock::now();
            preprocessLatencySum += end - begin;

            begin = system_clock::now();
            task.client->copyInput();
            end = system_clock::now();
            copyInputLatencySum += end - begin;

            begin = system_clock::now();
            task.client->infer();
            end = system_clock::now();
            inferLatencySum += end - begin;

            begin = system_clock::now();
            task.client->copyOutput();
            end = system_clock::now();
            copyOutputLatencySum += end - begin;

            begin = system_clock::now();
            task.client->postprocess();
            end = system_clock::now();
            postprocessLatencySum += end - begin;
        }

        task.standAloneLatency->prepareInputLatency = prepareInputLatencySum / LATENCY_MEASURE_TIME;
        task.standAloneLatency->preprocessLatency = preprocessLatencySum / LATENCY_MEASURE_TIME;
        task.standAloneLatency->copyInputLatency = copyInputLatencySum / LATENCY_MEASURE_TIME;
        task.standAloneLatency->inferLatency = inferLatencySum / LATENCY_MEASURE_TIME;
        task.standAloneLatency->copyOutputLatency = copyOutputLatencySum / LATENCY_MEASURE_TIME;
        task.standAloneLatency->postprocessLatency = postprocessLatencySum / LATENCY_MEASURE_TIME;
    }

    // 3. coordinate
    std::vector<StrategyInfo> strategyInfos;
    for (auto &task : tasks) {
        strategyInfos.emplace_back(task.strategy, task.standAloneLatency);
    }
    strategyCoordinator(strategyInfos);

    // 4. launch benchmark
    beginTime = system_clock::now() + seconds(DELAY_BEGIN_SECONDS);
    endTime = beginTime + seconds(benchmarkTime);

    std::vector<std::thread> benchmarkThreads;
    
    for (BenchmarkTask &task : tasks) {
        benchmarkThreads.emplace_back(&BenchmarkTask::runBenchmark, task, beginTime, endTime);
    }
    
    for (auto &thread : benchmarkThreads) {
        thread.join();
    }
}

Json::Value BenchmarkSuite::generateReport()
{
    Json::Value report;
    report["beginTime"] = timepointToString(beginTime);
    report["endTime"] = timepointToString(endTime);
    report["benchmarkTime(s)"] = (Json::Value::Int64) benchmarkTime.count();

    Json::Value results;
    for (auto &task : tasks) {
        results.append(task.client->generateReport());
    }
    report["results"] = results;

    return report;
}

}