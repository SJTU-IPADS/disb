#include "disb.h"
#include "rtclient.h"

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: rt_bench config.json" << std::endl;
        exit(-1);
    }

    std::string jsonStr = readStringFromFile(argv[1]);

    // show all pictures
    printf("showing input dataset...\n");
    Json::Value config;
    Json::Reader().parse(jsonStr, config, false);
    for (int i = 0; i < 10; ++i) {
        std::string inputDataPath = config["tasks"][0]["client"]["inputDataPath"].asString();
        cv::Mat img = cv::imread(inputDataPath + "/" + std::to_string(i) + ".png");
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
        cv::resize(img, img, cv::Size(28, 28));

        printf("\n%d.png:\n", i);
        for (int h = 0; h < 28; ++h) {
            for (int w = 0; w < 28; ++w) {
                if (img.data[h * 28 + w] > 200) {
                    printf("  ");
                } else {
                    printf("@ ");
                }
            }
            printf("\n");
        }
    }

    DISB::BenchmarkSuite benchmark;
    benchmark.init(jsonStr, rtClientFactory);
    benchmark.run();
    Json::Value report = benchmark.generateReport();
    std::cout << "\nreport:\n" << report << std::endl;
}