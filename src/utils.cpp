#include "utils.h"

#include <chrono>
#include <string>
#include <sstream>
#include <fstream>

std::string readStringFromFile(const std::string &filename)
{
    std::fstream file(filename);
    std::stringstream ss;
    ss << file.rdbuf();
    file.close();
    return ss.str();
}

std::string timepointToString(const std::chrono::system_clock::time_point &timepoint)
{
    char timeStr[25] = {0};
    time_t tt = std::chrono::system_clock::to_time_t(timepoint);
    struct tm localTime;
    localtime_r(&tt, &localTime);
    strftime(timeStr, 25, "%Y-%m-%d %H:%M:%S", &localTime);
 
    return std::string(timeStr);
}