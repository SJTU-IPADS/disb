/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _DISB_TENSORRT_UTILS_H_
#define _DISB_TENSORRT_UTILS_H_

#include <map>
#include <iostream>
#include <assert.h>
#include <NvInfer.h>
#include <jsoncpp/json/json.h>

constexpr long long int operator"" _MiB(unsigned long long val)
{
    return val * (1 << 20);
}

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

inline size_t volume(const nvinfer1::Dims& d)
{
    size_t vol = 1;
    const int32_t nbDims = d.nbDims;
    for (int i = 0; i < nbDims; i++) {
        int32_t dim = d.d[i];
        dim = dim == -1 ? 1 : dim;
        vol *= dim;
    }
    return vol;
}

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

inline void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0) {
        if (builder->getNbDLACores() == 0) {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores\n";
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback) {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8)) {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const { delete obj; }
};

class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

#endif
