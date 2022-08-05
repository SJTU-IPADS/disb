#include "disb.h"
#include "tfclient.h"

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: tf_benchmark config.json" << std::endl;
        exit(-1);
    }

    std::string jsonStr = readStringFromFile(argv[1]);

    DISB::BenchmarkSuite benchmark;
    benchmark.init(jsonStr, tfClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << "\nreport:\n" << report << std::endl;
}
