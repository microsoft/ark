// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_LOGGING_H_
#define ARK_GPU_LOGGING_H_

#include <cuda.h>

#include <sstream>

#include "logging.h"

#define CULOG(cmd)                                                \
    do {                                                          \
        CUresult _e = cmd;                                        \
        if (_e != CUDA_SUCCESS) {                                 \
            const char *_estr;                                    \
            cuGetErrorString(_e, &_estr);                         \
            LOG(ark::ERROR, "CUDA error ", _e, " '", _estr, "'"); \
        }                                                         \
    } while (0)

#define NVMLLOG(cmd)                                                      \
    do {                                                                  \
        nvmlReturn_t _e = cmd;                                            \
        if (_e != NVML_SUCCESS) {                                         \
            LOG(ark::ERROR, "NVML error ", _e, " '", nvmlErrorString(_e), \
                "'");                                                     \
        }                                                                 \
    } while (0)

#define NVRTCLOG(cmd)                                                          \
    do {                                                                       \
        nvrtcResult _e = cmd;                                                  \
        if (_e != NVRTC_SUCCESS) {                                             \
            LOG(ark::ERROR, "NVRTC error ", _e, " '", nvrtcGetErrorString(_e), \
                "'");                                                          \
        }                                                                      \
    } while (0)

#endif  // ARK_GPU_LOGGING_H_