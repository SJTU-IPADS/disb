#include "disb_utils.h"

#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <jsoncpp/json/json.h>

std::string readStringFromFile(const std::string &filename)
{
    std::fstream file(filename);
    std::stringstream ss;
    ss << file.rdbuf();
    file.close();
    return ss.str();
}

Json::Value readJsonFromFile(const std::string &filename)
{
    Json::Value v;
    Json::Reader reader;
    std::fstream file(filename);
    reader.parse(file, v, false);
    file.close();
    return v;
}

void writeJsonToFile(const std::string &filename, const Json::Value &json)
{
    std::ofstream file(filename);
    file << json;
    file.close();
}

std::string joinPath(std::string path1, std::string path2)
{
    if (path1.length() <= 0) {
        path1 = ".";
    } else if (path1[0] != '/' && path1[0] != '.') {
        path1 = "./" + path1;
    }

    // remove the last '/'
    if (path1[path1.length() - 1] == '/') {
        path1 = path1.substr(0, path1.length() - 1);
    }

    if (path2.length() <= 0) return path1;
    // add the last '/'
    if (path2[path2.length() - 1] != '/') {
        path2 = path2 + "/";
    }

    // process ".." & "../"
    while (path2.length() >= 2) {
        if (path2[0] == '.') {
            if (path2[1] == '.') {
                // ".."
                path2 = path2.substr(2);
                if (path2.length() > 0 && path2[0] == '/') {
                    path2 = path2.substr(1);
                }

                if (path1.length() == 1) {
                    if (path1[0] == '.') {
                        path1 = "..";
                    }
                    continue;
                }

                if (path1.length() >= 2 && path1[0] == '.' && path1[1] == '.') {
                    path1 += "/..";
                } else {
                    std::size_t sep_pos = path1.find_last_of('/');
                    if (sep_pos == 0) {
                        path1 = "/";
                    } else {
                        path1 = path1.substr(0, sep_pos);
                    }
                }
                continue;
            } else if (path2[1] == '/') {
                // "./"
                path2 = path2.substr(2);
                continue;
            }
        }
        break;
    }

    if (path2.length() == 0) return path1;
    if (path2[path2.length() - 1] == '/') {
        path2 = path2.substr(0, path2.length() - 1);
    }
    return path1 + "/" + path2;
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