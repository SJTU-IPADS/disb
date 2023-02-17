#ifndef _DISB_TRITON_ACCANALYZER_H_
#define _DISB_TRITON_ACCANALYZER_H_

#include "disb.h"

#include <memory>

class AccuarcyAnalyzer: public DISB::Analyzer
{
private:
    int totalInference;
    int correctInference;

public:
    AccuarcyAnalyzer();
    virtual void start(const std::chrono::system_clock::time_point &beginTime) override;
    
    virtual std::shared_ptr<DISB::Record> produceRecord() override;
    virtual void consumeRecord(std::shared_ptr<DISB::Record> record) override;
    virtual Json::Value generateReport() override;
};

#endif
