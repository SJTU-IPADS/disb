#ifndef _DISB_UTILS_H_
#define _DISB_UTILS_H_

#include <string>
#include <chrono>
#include <jsoncpp/json/json.h>

#define THOUSAND 1000
#define BILLION 1000000000

std::string readStringFromFile(const std::string &filename);
Json::Value readJsonFromFile(const std::string &filename);
void writeJsonToFile(const std::string &filename, const Json::Value &json);
std::string joinPath(std::string path1, std::string path2);
std::string timepointToString(const std::chrono::system_clock::time_point &timepoint);

#endif
