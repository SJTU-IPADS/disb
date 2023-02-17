#include "load.h"
#include "disb_utils.h"

#include <map>
#include <cmath>
#include <vector>
#include <chrono>
#include <memory>
#include <thread>
#include <cassert>
#include <algorithm>
#include <jsoncpp/json/json.h>

namespace DISB
{

using std::chrono::system_clock;
using std::chrono::milliseconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;

void builtinLoadCoordinator(const std::vector<LoadInfo> &loadInfos)
{
    // if two or more periodic strategies has the same frequency and priority is 0
    // they need to coordinate by setting launchDelay to avoid being launched at the same time
    std::map<int, std::chrono::nanoseconds> delayMap;
    for (auto &loadInfo : loadInfos) {
        std::shared_ptr<PeriodicLoad> periodicLoad = std::dynamic_pointer_cast<PeriodicLoad>(loadInfo.load);
        if (periodicLoad == nullptr) {
            continue;
        }

        int priority = periodicLoad->getPriority();
        if (priority > 0) {
            continue;
        }

        nanoseconds delay = loadInfo.standAloneLatency->sum();

        int freq = periodicLoad->getFrequency();
        periodicLoad->setLaunchDelay(delayMap[freq]);
        delayMap[freq] += delay;
    }
}

std::shared_ptr<Load> builtinLoadFactory(const Json::Value &config)
{
    std::string type = config["type"].asString();

    if (type == "continuous") {
        return std::make_shared<ContinuousLoad>();
    } else if (type == "periodic") {
        return std::make_shared<PeriodicLoad>(config);
    } else if (type == "poisson") {
        return std::make_shared<PoissonLoad>(config);
    } else if (type == "trace") {
        return std::make_shared<TraceLoad>(config);
    } else if (type == "dependent") {
        return std::make_shared<DependentLoad>(config);
    }

    printf("ERROR: unknown load type: %s\n", type.c_str());
    return nullptr;
}

void ContinuousLoad::start(const std::chrono::system_clock::time_point &_beginTime,
                           const std::chrono::system_clock::time_point &_endTime)
{
    beginTime = _beginTime;
    endTime = _endTime;
}

bool ContinuousLoad::waitUntilNextLaunch()
{
    auto now = system_clock::now();
    if (now < beginTime) std::this_thread::sleep_until(beginTime);
    return now < endTime;
}

PeriodicLoad::PeriodicLoad(const Json::Value &config):
    priority(0), frequency(0), interval(0), launchDelay(0)
{
    priority = config["priority"].asInt();
    frequency = config["frequency"].asFloat();

    if (frequency != 0) {
        interval = nanoseconds(int64_t(BILLION / frequency));
    }
}

int PeriodicLoad::getPriority()
{
    return priority;
}

int PeriodicLoad::getFrequency()
{
    return frequency;
}

void PeriodicLoad::setLaunchDelay(const std::chrono::nanoseconds &_launchDelay)
{
    // printf("set launch delay: %ldus\n", _launchDelay.count() / 1000);
    launchDelay = _launchDelay;
}

void PeriodicLoad::start(const std::chrono::system_clock::time_point &_beginTime,
                         const std::chrono::system_clock::time_point &_endTime)
{
    if (frequency == 0) {
        nextLaunch = system_clock::time_point::max();
    } else {
        nextLaunch = _beginTime + launchDelay;
    }
    endTime = _endTime;
}

bool PeriodicLoad::waitUntilNextLaunch()
{
// Current time `now` may be behind `nextLaunch`, in other words, `nextLaunch`
// has already arrived. In that case, DISB will check whether the `nextNextLaunch`
// has arrived. If the `nextNextLaunch` has arrived, DISB will give up the `nextLaunch`.

// This simulates real workloads better. For example, in object detection,
// the new frames arrive periodically. To ensure timeliness,
// the DNN model should always choose the newest frame as input.
// In that case (case 2), if the nextFrame (or `nextLaunch`) and
// the nextNextFrame (or `nextNextLaunch`) has both arrived,
// the DNN model will give up the nextFrame and choose the nextNextFrame.

// case 1: still launch at `nextLaunch`
//      nextLaunch        now    nextNextLaunch
//  ---------|-------------|-----------|--------

// case 2: give up `nextLaunch`
//      nextLaunch      nextNextLaunch       now
//  ---------|----------------|---------------|--------

    system_clock::time_point now = system_clock::now();
    system_clock::time_point nextNextLaunch = nextLaunch + interval;
    
    while (nextNextLaunch < now) {
        nextLaunch = nextNextLaunch;
        nextNextLaunch = nextLaunch + interval;
    }

    if (nextLaunch >= endTime) return false;

    std::this_thread::sleep_until(nextLaunch);
    nextLaunch = nextNextLaunch;
    return true;
}

PoissonLoad::PoissonLoad(const Json::Value &config)
    : uniformDistribution(0, 1)
{
    frequency = config["frequency"].asDouble();
}

void PoissonLoad::start(const std::chrono::system_clock::time_point &_beginTime,
                        const std::chrono::system_clock::time_point &_endTime)
{
    nextLaunch = _beginTime;
    endTime = _endTime;
    randomEngine.seed(system_clock::now().time_since_epoch().count());
}

bool PoissonLoad::waitUntilNextLaunch()
{
// Current time `now` may be behind `nextLaunch`, in other words, `nextLaunch`
// has already arrived. In that case, DISB will check whether the `nextNextLaunch`
// has arrived. If the `nextNextLaunch` has arrived, DISB will give up the `nextLaunch`.

// This simulates real workloads better. For example, in object detection,
// the new frames arrive in poisson distribution. To ensure timeliness,
// the DNN model should always choose the newest frame as input.
// In that case (case 2), if the nextFrame (or `nextLaunch`) and
// the nextNextFrame (or `nextNextLaunch`) has both arrived,
// the DNN model will give up the nextFrame and choose the nextNextFrame.

// case 1: still launch at `nextLaunch`
//      nextLaunch        now    nextNextLaunch
//  ---------|-------------|-----------|--------

// case 2: give up `nextLaunch`
//      nextLaunch      nextNextLaunch       now
//  ---------|----------------|---------------|--------

    if (frequency <= 0) return false;

    system_clock::time_point now = system_clock::now();
    auto nextNextLaunch = nextLaunch + expRandInterval();

    while (nextNextLaunch < now) {
        nextLaunch = nextNextLaunch;
        nextNextLaunch = nextLaunch + expRandInterval();
    }
    if (nextLaunch >= endTime) return false;
    
    std::this_thread::sleep_until(nextLaunch);
    nextLaunch = nextNextLaunch;
    return true;
}

microseconds PoissonLoad::expRandInterval()
{
    // The time interval between two events in the Poisson process
    // satisfies the exponential distribution.
    double u = uniformDistribution(randomEngine);
    double interval = (-1/frequency) * log(u);
    return microseconds(int64_t(interval * THOUSAND * THOUSAND));
}

TraceLoad::TraceLoad(const Json::Value &config): nextTrace(0)
{
    if (config["trace"].isArray()) {
        for (const Json::Value &t : config["trace"]) {
            trace.push_back(t.asInt());
        }
    }

    std::sort(trace.begin(), trace.end());
}

void TraceLoad::start(const std::chrono::system_clock::time_point &_beginTime,
                      const std::chrono::system_clock::time_point &_endTime)
{
    nextTrace = 0;
    beginTime = _beginTime;
    endTime = _endTime;
}

bool TraceLoad::waitUntilNextLaunch()
{
// Current time `now` may be behind `nextLaunch`, in other words, `nextLaunch`
// has already arrived. In that case, DISB will check whether the `nextNextLaunch`
// has arrived. If the `nextNextLaunch` has arrived, DISB will give up the `nextLaunch`.

// This simulates real workloads better. For example, in object detection,
// the trace is the time when new frames arrive. To ensure timeliness,
// the DNN model should always choose the newest frame as input.
// In that case (case 2), if the nextFrame (or `nextLaunch`) and
// the nextNextFrame (or `nextNextLaunch`) has both arrived,
// the DNN model will give up the nextFrame and choose the nextNextFrame.

// case 1: still launch at `nextLaunch`
//      nextLaunch        now    nextNextLaunch
//  ---------|-------------|-----------|--------

// case 2: give up `nextLaunch`
//      nextLaunch      nextNextLaunch       now
//  ---------|----------------|---------------|--------

    system_clock::time_point now = system_clock::now();
    if (nextTrace >= trace.size()) return false;

    size_t nextNextTrace = nextTrace + 1;
    system_clock::time_point nextLaunch = beginTime + milliseconds(trace[nextTrace]);
    while (nextNextTrace < trace.size()) {
        system_clock::time_point nextNextLaunch = beginTime + milliseconds(trace[nextNextTrace]);
        if (nextNextLaunch >= now) break;

        nextLaunch = nextNextLaunch;
        nextNextTrace += 1;
    }
    nextTrace = nextNextTrace;

    if (nextLaunch >= endTime) return false;

    std::this_thread::sleep_until(nextLaunch);
    return true;
}

DependentLoad::DependentLoad(const Json::Value &config)
{
    if (config["prev_task_ids"].isArray()) {
        for (const Json::Value &prev : config["prev_task_ids"]) {
            prevTaskIds.push_back(prev.asString());
        }
    }
}

void DependentLoad::start(const std::chrono::system_clock::time_point &,
                          const std::chrono::system_clock::time_point &_endTime)
{
    endTime = _endTime;
}

bool DependentLoad::waitUntilNextLaunch()
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait_until(lock, endTime, [&]{
        bool ready = false;
        for (auto &res : prevResults) {
            if (res.second.size() == prevTaskIds.size()) {
                // all previous result of one serial number have arrived
                ready = true;
                break;
            }
        }
        return ready;
    });
    return system_clock::now() < endTime;
}

void DependentLoad::recvResult(std::shared_ptr<InferResult> inferResult)
{
    mtx.lock();
    prevResults[inferResult->resultSerialNumber][inferResult->producerTaskId] = inferResult;

    // all previous result of the SerialNumber have arrived
    bool ready = prevResults[inferResult->resultSerialNumber].size() == prevTaskIds.size();
    mtx.unlock();
    if (ready) cv.notify_all();
}

const std::vector<std::string> &DependentLoad::getPrevTaskIds() const
{
    return prevTaskIds;
}

std::map<std::string, std::shared_ptr<InferResult>> DependentLoad::getPrevResults()
{
    std::unique_lock<std::mutex> lock(mtx);

    // give up results before the first serial number with all its results arrived
    while (prevResults.size() > 0 && prevResults.begin()->second.size() < prevTaskIds.size()) {
        prevResults.erase(prevResults.begin());
    }

    assert(prevResults.size() > 0);
    std::map<std::string, std::shared_ptr<InferResult>> results = prevResults.begin()->second;
    prevResults.erase(prevResults.begin());

    return results;
}

} // namespace DISB
