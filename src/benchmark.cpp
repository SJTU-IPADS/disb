#include "benchmark.h"
#include "disb_utils.h"

#include <vector>
#include <thread>
#include <memory>
#include <cassert>
#include <iostream>
#include <jsoncpp/json/json.h>

namespace DISB
{

using std::chrono::system_clock;
using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

BenchmarkTask::BenchmarkTask(std::string taskId, std::shared_ptr<Client> clt, std::shared_ptr<Load> stg):
    id(taskId), client(clt), load(stg), standAloneLatency(std::make_shared<StandAloneLatency>())
{
    std::shared_ptr<DependentLoad> dependentLoad = std::dynamic_pointer_cast<DependentLoad>(load);
    if (dependentLoad == nullptr) return;

    isDependent = true;
    std::shared_ptr<DependentClient> dependentClient = std::dynamic_pointer_cast<DependentClient>(client);

    if (dependentClient == nullptr) {
        std::cerr << "[Error]: Client " << id << " has a dependent load, it should inherit DISB::DependentClient\n";
        exit(-1);
    }

    // check every dependent task should have at least one previous task
    if (dependentLoad->getPrevTaskIds().size() == 0) {
        std::cerr << "[Error]: DependentLoad of Client " << id << " has no dependencies\n";
        exit(-1);
    }
}

void BenchmarkTask::inferOnce()
{
    if (isDependent) {
        std::shared_ptr<DependentLoad> dependentLoad = std::dynamic_pointer_cast<DependentLoad>(load);
        std::shared_ptr<DependentClient> dependentClient = std::dynamic_pointer_cast<DependentClient>(client);
        assert(dependentLoad);
        assert(dependentClient);

        dependentClient->consumePrevResults(dependentLoad->getPrevResults());
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

    std::shared_ptr<InferResult> inferResult = client->produceResult();
    inferResult->producerTaskId = this->id;

    for (auto nextTask : nextTasks) {
        std::shared_ptr<DependentLoad> dependentLoad = std::dynamic_pointer_cast<DependentLoad>(nextTask->load);
        if (dependentLoad == nullptr) {
            continue;
        }
        inferResult->consumerTaskId = nextTask->id;
        dependentLoad->recvResult(inferResult);
    }
}

void BenchmarkTask::runBenchmark(const system_clock::time_point &beginTime,
                                 const system_clock::time_point &endTime)
{
    load->start(beginTime, endTime);
    client->startAnalyzers(beginTime);
    
    while (load->waitUntilNextLaunch()) {
        inferOnce();
    }

    load->stop();
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
                          std::shared_ptr<Load> loadFactory(const Json::Value &config))
{
    Json::Reader reader;
    Json::Value configJson;

    reader.parse(configJsonStr, configJson, false);
    init(configJson, clientFactory, loadFactory);
}

void BenchmarkSuite::init(const Json::Value &configJson,
                          std::shared_ptr<Client> clientFactory(const Json::Value &config),
                          std::shared_ptr<Load> loadFactory(const Json::Value &config))
{
    benchmarkTime = seconds(configJson["time"].asInt());
    for (const Json::Value &taskConfig : configJson["tasks"]) {
        std::string taskId = taskConfig["id"].asString();
        std::shared_ptr<Client> client = clientFactory(taskConfig["client"]);
        std::shared_ptr<Load> load = loadFactory(taskConfig["load"]);

        // check if the factory method is correct
        if (client != nullptr && std::dynamic_pointer_cast<Client>(client) != nullptr
            && load != nullptr && std::dynamic_pointer_cast<Load>(load) != nullptr) {
            
            client->init();
            client->initAnalyzers();
            load->init();
            tasks[taskId] = BenchmarkTask(taskId, client, load);
        }
    }

    for (auto &taskPair : tasks) {
        auto &task = taskPair.second;
        std::shared_ptr<DependentLoad> dependentLoad = std::dynamic_pointer_cast<DependentLoad>(task.load);
        if (dependentLoad == nullptr) continue;

        for (auto &prevTaskId : dependentLoad->getPrevTaskIds()) {
            if (tasks.find(prevTaskId) == tasks.end()) {
                std::cerr << "[Error]: Task ID " << prevTaskId << " NOT FOUND\n";
                exit(-1);
            }
            tasks[prevTaskId].nextTasks.push_back(&task);
        }
    }
}

void BenchmarkSuite::run(void loadCoordinator(const std::vector<LoadInfo> &loadInfos))
{
    // 1. warm up
    for (auto &taskPair : tasks) {
        auto &task = taskPair.second;

        if (task.isDependent) {
            std::shared_ptr<DependentClient> dependentClient = std::dynamic_pointer_cast<DependentClient>(task.client);
            assert(dependentClient);
            dependentClient->consumePrevResults(dependentClient->produceDummyPrevResults());
        }

        for (int i = 0; i < WARMUP_TIME; ++ i) {
            task.client->prepareInput();
            task.client->preprocess();
            task.client->copyInput();
            task.client->infer();
            task.client->copyOutput();
            task.client->postprocess();
        }
    }

    // 2. measure the stand alone latency of each client
    for (auto &taskPair : tasks) {
        auto &task = taskPair.second;
        // sleep 100 ms to reduce influence between two tasks
        std::this_thread::sleep_for(microseconds(100000));

        nanoseconds prepareInputLatencySum(0);
        nanoseconds preprocessLatencySum(0);
        nanoseconds copyInputLatencySum(0);
        nanoseconds inferLatencySum(0);
        nanoseconds copyOutputLatencySum(0);
        nanoseconds postprocessLatencySum(0);

        for (int i = 0; i < STANDALONE_LATENCY_MEASURE_TIME; ++i) {
            if (task.isDependent) {
                std::shared_ptr<DependentClient> dependentClient = std::dynamic_pointer_cast<DependentClient>(task.client);
                assert(dependentClient);
                dependentClient->consumePrevResults(dependentClient->produceDummyPrevResults());
            }
        
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

        task.standAloneLatency->prepareInputLatency = prepareInputLatencySum / STANDALONE_LATENCY_MEASURE_TIME;
        task.standAloneLatency->preprocessLatency = preprocessLatencySum / STANDALONE_LATENCY_MEASURE_TIME;
        task.standAloneLatency->copyInputLatency = copyInputLatencySum / STANDALONE_LATENCY_MEASURE_TIME;
        task.standAloneLatency->inferLatency = inferLatencySum / STANDALONE_LATENCY_MEASURE_TIME;
        task.standAloneLatency->copyOutputLatency = copyOutputLatencySum / STANDALONE_LATENCY_MEASURE_TIME;
        task.standAloneLatency->postprocessLatency = postprocessLatencySum / STANDALONE_LATENCY_MEASURE_TIME;

        task.client->setStandAloneLatency(task.standAloneLatency->sum());
    }

    // 3. coordinate
    std::vector<LoadInfo> loadInfos;
    for (auto &taskPair : tasks) {
        auto &task = taskPair.second;
        loadInfos.emplace_back(task.load, task.standAloneLatency);
    }
    loadCoordinator(loadInfos);

    // 4. launch benchmark
    beginTime = system_clock::now() + seconds(DELAY_BEGIN_SECONDS);
    endTime = beginTime + seconds(benchmarkTime);

    std::vector<std::thread> benchmarkThreads;
    
    for (auto &taskPair : tasks) {
        auto &task = taskPair.second;
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
    for (auto &taskPair : tasks) {
        results.append(taskPair.second.client->generateReport());
    }
    report["results"] = results;

    return report;
}

}