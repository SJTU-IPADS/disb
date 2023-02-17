#include "disb.h"
#include "tfclient.h"

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: tfs_benchmark [benchmarks dir] [workload name]" << std::endl;
        exit(-1);
    }

    std::string benchmarksDir(argv[1]);
    std::string workloadPath = joinPath(joinPath(benchmarksDir, "workloads"), std::string(argv[2]) + ".json");

    Json::Value config = readJsonFromFile(workloadPath);
    for (auto &taskConfig : config["tasks"]) {
        taskConfig["client"]["benchmarks_dir"] = benchmarksDir;
    }

    DISB::BenchmarkSuite benchmark;
    benchmark.init(config, tfClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << "\nreport:\n" << report << std::endl;
    std::string resultPath = joinPath(joinPath(benchmarksDir, "results"), std::string("tfs.") + argv[2] + ".json");
    writeJsonToFile(resultPath, report);
}
