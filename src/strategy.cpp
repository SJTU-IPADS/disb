#include "strategy.h"
#include "utils.h"

#include <map>
#include <vector>
#include <chrono>
#include <memory>
#include <jsoncpp/json/json.h>

namespace DISB
{

using std::chrono::system_clock;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

void builtinStrategyCoordinator(const std::vector<StrategyInfo> &strategyInfos)
{
    // if two or more periodic strategies has the same frequency and priority is 0
    // they need to coordinate by setting launchDelay to avoid being launched at the same time
    std::map<int, std::chrono::nanoseconds> delayMap;
    for (auto &strategyInfo : strategyInfos) {
        PeriodicStrategy *periodicStrategy = dynamic_cast<PeriodicStrategy *>(strategyInfo.strategy.get());
        if (periodicStrategy == nullptr) {
            continue;
        }

        int priority = periodicStrategy->getPriority();
        if (priority > 0) {
            continue;
        }

        nanoseconds delay(0);
        delay += strategyInfo.standAloneLatency->prepareInputLatency;
        delay += strategyInfo.standAloneLatency->preprocessLatency;
        delay += strategyInfo.standAloneLatency->copyInputLatency;
        delay += strategyInfo.standAloneLatency->inferLatency;
        delay += strategyInfo.standAloneLatency->copyOutputLatency;
        delay += strategyInfo.standAloneLatency->postprocessLatency;

        int freq = periodicStrategy->getFrequency();
        periodicStrategy->setLaunchDelay(delayMap[freq]);
        delayMap[freq] += delay;
    }
}

std::shared_ptr<Strategy> builtinStrategyFactory(const Json::Value &config)
{
    std::string type = config["type"].asString();

    if (type == "periodic") {
        return std::make_shared<PeriodicStrategy>(config);
    } else if (type == "trace") {
        return std::make_shared<TraceStrategy>(config);
    }

    return nullptr;
}

PeriodicStrategy::PeriodicStrategy(const Json::Value &config):
    priority(0), frequency(0), interval(0), launchDelay(0)
{
    priority = config["priority"].asInt();
    frequency = config["frequency"].asFloat();

    if (frequency != 0) {
        interval = nanoseconds(int64_t(BILLION / frequency));
    }
}

int PeriodicStrategy::getPriority()
{
    return priority;
}

int PeriodicStrategy::getFrequency()
{
    return frequency;
}

void PeriodicStrategy::setLaunchDelay(const std::chrono::nanoseconds &_launchDelay)
{
    launchDelay = _launchDelay;
}

void PeriodicStrategy::start(const system_clock::time_point &beginTime)
{
    if (frequency == 0) {
        nextLaunch = system_clock::time_point::max();
    } else {
        nextLaunch = beginTime + launchDelay;
    }
}

system_clock::time_point PeriodicStrategy::nextLaunchTime(const system_clock::time_point &now)
{
    system_clock::time_point nextNextLaunch = nextLaunch + interval;
    
    while (nextNextLaunch < now) {
        nextLaunch = nextNextLaunch;
        nextNextLaunch = nextLaunch + interval;
    }

    system_clock::time_point retTime = nextLaunch;
    nextLaunch += interval;
    return retTime;
}

TraceStrategy::TraceStrategy(const Json::Value &config): nextTrace(0)
{
    if (config["trace"].isArray()) {
        Json::Value configTrace = config["trace"];
        for (auto t : configTrace) {
            trace.push_back(t.asInt());
        }
    }
}

void TraceStrategy::start(const system_clock::time_point &beginTime)
{
    nextTrace = 0;
    firstLaunchTime = beginTime;
}

system_clock::time_point TraceStrategy::nextLaunchTime(const system_clock::time_point &now)
{
    if (nextTrace >= trace.size()) {
        return system_clock::time_point::max();
    }
    
    size_t nextNextTrace = nextTrace + 1;
    system_clock::time_point nextLaunch = firstLaunchTime + milliseconds(trace[nextTrace]);
    while (nextNextTrace < trace.size()) {
        system_clock::time_point nextNextLaunch = firstLaunchTime + milliseconds(trace[nextNextTrace]);
        if (nextNextLaunch >= now) {
            break;
        }
        nextLaunch = nextNextLaunch;
        nextNextTrace += 1;
    }
    nextTrace = nextNextTrace;
    return nextLaunch;
}

} // namespace DISB
