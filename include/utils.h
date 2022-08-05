#ifndef _DISB_UTILS_H_
#define _DISB_UTILS_H_

#include <string>
#include <chrono>

#define THOUSAND 1000
#define BILLION 1000000000

std::string readStringFromFile(const std::string &filename);
std::string timepointToString(const std::chrono::system_clock::time_point &timepoint);

#endif
