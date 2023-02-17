#ifndef _DISB_ANALYZER_H_
#define _DISB_ANALYZER_H_

#include <list>
#include <chrono>
#include <memory>
#include <jsoncpp/json/json.h>

namespace DISB {

class Analyzer;
class Client;

struct TimePoints
{
    std::chrono::system_clock::time_point prepareInputBegin;
    std::chrono::system_clock::time_point prepareInputEnd;

    std::chrono::system_clock::time_point preprocessBegin;
    std::chrono::system_clock::time_point preprocessEnd;

    std::chrono::system_clock::time_point copyInputBegin;
    std::chrono::system_clock::time_point copyInputEnd;

    std::chrono::system_clock::time_point inferBegin;
    std::chrono::system_clock::time_point inferEnd;

    std::chrono::system_clock::time_point copyOutputBegin;
    std::chrono::system_clock::time_point copyOutputEnd;

    std::chrono::system_clock::time_point postprocessBegin;
    std::chrono::system_clock::time_point postprocessEnd;
};

class Record
{
public:
    // avoid recursive dependency in shared_ptr
    Analyzer *analyzer;
    std::shared_ptr<TimePoints> timePoints;
};

class Analyzer
{
private:
    Client *client;
    
public:
    Analyzer();

    Client *getClient();
    void setClient(Client *clt);

    virtual void init() {}
    virtual void start(const std::chrono::system_clock::time_point &beginTime) {}
    virtual void stop(const std::chrono::system_clock::time_point &endTime) {}
    
    std::shared_ptr<Record> produceRecordWrapper();
    virtual std::shared_ptr<Record> produceRecord() = 0;
    virtual void consumeRecord(std::shared_ptr<Record> record) {}
    virtual Json::Value generateReport() = 0;

    virtual void onPrepareInput(std::shared_ptr<Record> record) {}
    virtual void onPreprocess(std::shared_ptr<Record> record) {}
    virtual void onCopyInput(std::shared_ptr<Record> record) {}
    virtual void onInfer(std::shared_ptr<Record> record) {}
    virtual void onCopyOutput(std::shared_ptr<Record> record) {}
    virtual void onPostprocess(std::shared_ptr<Record> record) {}
};

class BasicAnalyzer: public Analyzer
{
private:
    std::chrono::nanoseconds standAloneLatency;
    std::list<std::shared_ptr<Record>> records;
    std::chrono::system_clock::time_point beginTime;
    std::chrono::system_clock::time_point endTime;

public:
    void setStandAloneLatency(std::chrono::nanoseconds standAloneLatency);

    virtual void start(const std::chrono::system_clock::time_point &beginTime) override;
    virtual void stop(const std::chrono::system_clock::time_point &endTime) override;
    
    virtual std::shared_ptr<Record> produceRecord() override;
    virtual void consumeRecord(std::shared_ptr<Record> record) override;
    virtual Json::Value generateReport() override;
};

} // namespace DISB

#endif
