#ifndef _DISB_LOAD_H_
#define _DISB_LOAD_H_

#include "client.h"

#include <map>
#include <vector>
#include <chrono>
#include <memory>
#include <random>
#include <mutex>
#include <condition_variable>
#include <jsoncpp/json/json.h>

namespace DISB
{

struct StandAloneLatency
{
    std::chrono::nanoseconds prepareInputLatency;
    std::chrono::nanoseconds preprocessLatency;
    std::chrono::nanoseconds copyInputLatency;
    std::chrono::nanoseconds inferLatency;
    std::chrono::nanoseconds copyOutputLatency;
    std::chrono::nanoseconds postprocessLatency;

    StandAloneLatency():
        prepareInputLatency(0),
        preprocessLatency(0),
        copyInputLatency(0),
        inferLatency(0),
        copyOutputLatency(0),
        postprocessLatency(0) {}
    
    std::chrono::nanoseconds sum()
    {
        std::chrono::nanoseconds latencySum(0);
        latencySum += prepareInputLatency;
        latencySum += preprocessLatency;
        latencySum += copyInputLatency;
        latencySum += inferLatency;
        latencySum += copyOutputLatency;
        latencySum += postprocessLatency;
        return latencySum;
    }
};

class Load
{
public:
    virtual void init() {}
    virtual void start(const std::chrono::system_clock::time_point &beginTime,
                       const std::chrono::system_clock::time_point &endTime) {}
    virtual void stop() {}
    virtual bool waitUntilNextLaunch() = 0;
};

struct LoadInfo
{
    std::shared_ptr<Load> load;
    std::shared_ptr<StandAloneLatency> standAloneLatency;

    LoadInfo(std::shared_ptr<Load> _load, std::shared_ptr<StandAloneLatency> _standAloneLatency):
        load(_load), standAloneLatency(_standAloneLatency) {}
};

// if two or more periodic loads has the same frequency and priority is 0
// they need to coordinate by setting launchDelay to avoid being launched at the same time
void builtinLoadCoordinator(const std::vector<LoadInfo> &loadInfos);
std::shared_ptr<Load> builtinLoadFactory(const Json::Value &config);

// builtin strategies

class ContinuousLoad: public Load
{
private:
    std::chrono::system_clock::time_point beginTime;
    std::chrono::system_clock::time_point endTime;

public:
    ContinuousLoad() {}
    virtual void start(const std::chrono::system_clock::time_point &beginTime,
                       const std::chrono::system_clock::time_point &endTime) override;
    virtual bool waitUntilNextLaunch() override;
};

class PeriodicLoad: public Load
{
private:
    int priority;
    float frequency;
    std::chrono::nanoseconds interval;
    std::chrono::nanoseconds launchDelay;
    std::chrono::system_clock::time_point nextLaunch;
    std::chrono::system_clock::time_point endTime;

public:
    PeriodicLoad(const Json::Value &config);

    int getPriority();
    int getFrequency();
    void setLaunchDelay(const std::chrono::nanoseconds &launchDelay);

    virtual void start(const std::chrono::system_clock::time_point &beginTime,
                       const std::chrono::system_clock::time_point &endTime) override;
    virtual bool waitUntilNextLaunch() override;
};

class PoissonLoad: public Load
{
private:
    double frequency;
    std::chrono::system_clock::time_point endTime;
    std::chrono::system_clock::time_point nextLaunch;
    
    std::default_random_engine randomEngine;
    std::uniform_real_distribution<double> uniformDistribution;

public:
    PoissonLoad(const Json::Value &config);
    virtual void start(const std::chrono::system_clock::time_point &beginTime,
                       const std::chrono::system_clock::time_point &endTime) override;
    virtual bool waitUntilNextLaunch() override;
    std::chrono::microseconds expRandInterval();
};

class TraceLoad: public Load
{
private:
    size_t nextTrace;
    std::vector<int> trace;
    std::chrono::system_clock::time_point beginTime;
    std::chrono::system_clock::time_point endTime;

public:
    TraceLoad(const Json::Value &config);
    virtual void start(const std::chrono::system_clock::time_point &beginTime,
                       const std::chrono::system_clock::time_point &endTime) override;
    virtual bool waitUntilNextLaunch() override;
};

class DependentLoad: public Load
{
private:
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::string> prevTaskIds;

    // prevResults: a map that stores all results from previous tasks
    // key: serial number (int64)
    // val: results map (map[string] => InferResult *)
    //
    // results map: a map that stores all results with the same serial number
    // key: task id of the previous task (string)
    // val: infer result of the previous task (InferResult *)
    std::map<int64_t, std::map<std::string, std::shared_ptr<InferResult>>> prevResults;
    std::chrono::system_clock::time_point endTime;
    
public:
    DependentLoad(const Json::Value &config);
    virtual void start(const std::chrono::system_clock::time_point &beginTime,
                       const std::chrono::system_clock::time_point &endTime) override;
    virtual bool waitUntilNextLaunch() override;

    void recvResult(std::shared_ptr<InferResult> inferResult);
    const std::vector<std::string> &getPrevTaskIds() const;
    virtual std::map<std::string, std::shared_ptr<InferResult>> getPrevResults();
};

} // namespace DISB

#endif
