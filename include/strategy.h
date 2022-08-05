#ifndef _DISB_STRATEGY_H_
#define _DISB_STRATEGY_H_

#include <vector>
#include <chrono>
#include <memory>
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
};

class Strategy
{
public:
    virtual void init() {}
    virtual void start(const std::chrono::system_clock::time_point &beginTime) {}
    virtual void stop(const std::chrono::system_clock::time_point &endTime) {}
    virtual std::chrono::system_clock::time_point nextLaunchTime(const std::chrono::system_clock::time_point &now) = 0;
};

struct StrategyInfo
{
    std::shared_ptr<Strategy> strategy;
    std::shared_ptr<StandAloneLatency> standAloneLatency;

    StrategyInfo(std::shared_ptr<Strategy> _strategy, std::shared_ptr<StandAloneLatency> _standAloneLatency):
        strategy(_strategy), standAloneLatency(_standAloneLatency) {}
};

// if two or more periodic strategy has the same frequency and priority is 0
// they need to coordinate by setting launchDelay to avoid being launched at the same time
void builtinStrategyCoordinator(const std::vector<StrategyInfo> &strategyInfos);
std::shared_ptr<Strategy> builtinStrategyFactory(const Json::Value &config);

// builtin strategies
class PeriodicStrategy: public Strategy
{
private:
    int priority;
    float frequency;
    std::chrono::nanoseconds interval;
    std::chrono::nanoseconds launchDelay;
    std::chrono::system_clock::time_point nextLaunch;

public:
    PeriodicStrategy(const Json::Value &config);

    int getPriority();
    int getFrequency();
    void setLaunchDelay(const std::chrono::nanoseconds &launchDelay);

    virtual void start(const std::chrono::system_clock::time_point &beginTime) override;
    virtual std::chrono::system_clock::time_point nextLaunchTime(const std::chrono::system_clock::time_point &now) override;
};

class TraceStrategy: public Strategy
{
private:
    size_t nextTrace;
    std::vector<int> trace;
    std::chrono::system_clock::time_point firstLaunchTime;

public:
    TraceStrategy(const Json::Value &config);
    virtual void start(const std::chrono::system_clock::time_point &beginTime) override;
    virtual std::chrono::system_clock::time_point nextLaunchTime(const std::chrono::system_clock::time_point &now) override;
};

} // namespace DISB

#endif
