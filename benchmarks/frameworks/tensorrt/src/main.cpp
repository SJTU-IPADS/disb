#include "disb.h"
#include "rtclient.h"

#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: trt_benchmark [benchmarks dir] [workload name]" << std::endl;
        exit(-1);
    }

    std::string benchmarksDir(argv[1]);
    std::string workloadPath = joinPath(joinPath(benchmarksDir, "workloads"), std::string(argv[2]) + ".json");
    
    size_t freeByte;
	size_t totalByte;
    if (cudaMemGetInfo(&freeByte, &totalByte) != cudaSuccess) {
        std::cout << "Error: Cannot get cuda memory info.\n";
        exit(-1);
    }

    std::map<std::string, int> modelNames;
    Json::Value config = readJsonFromFile(workloadPath);
    for (Json::Value &taskConfig : config["tasks"]) {
        int profileIdx = modelNames[taskConfig["client"]["model_name"].asString()];
        modelNames[taskConfig["client"]["model_name"].asString()] += 1;
        taskConfig["client"]["profile_idx"] = (Json::Value::Int)profileIdx;
    }
    size_t modelCnt = modelNames.size();
    size_t memoryLimit = ((freeByte >> 20) - 512) / modelCnt;
    for (auto &taskConfig : config["tasks"]) {
        taskConfig["client"]["benchmarks_dir"] = benchmarksDir;
        taskConfig["client"]["memory_limit"] = (Json::Value::Int64)memoryLimit;
        taskConfig["client"]["profile_cnt"] = (Json::Value::Int)modelNames[taskConfig["client"]["model_name"].asString()];
    }

    DISB::BenchmarkSuite *benchmark = new DISB::BenchmarkSuite();
    benchmark->init(config, rtClientFactory);
    benchmark->run();
    Json::Value report = benchmark->generateReport();
    
    delete benchmark;
    TensorRTClient::freeEngines();

    std::cout << "\nreport:\n" << report << std::endl;
    std::string resultPath = joinPath(joinPath(benchmarksDir, "results"), std::string("trt.") + argv[2] + ".json");
    writeJsonToFile(resultPath, report);
}