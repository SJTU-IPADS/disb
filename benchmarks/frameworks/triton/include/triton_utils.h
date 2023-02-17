#ifndef _DISB_TRITON_TRITON_UTILS_H_
#define _DISB_TRITON_TRITON_UTILS_H_

#include <string>
#include <cstdio>
#include <cstdlib>
#include "common.h"

#define ASSERT_TRITON_ERROR(cmd)\
{\
    triton::client::Error error = cmd;\
    if (!error.IsOk()) {\
        fprintf(stderr, "[ERR] triton client error: %s at %s:%d\n", error.Message().c_str(), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }\
}

#endif