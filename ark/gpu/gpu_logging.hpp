// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_LOGGING_HPP_
#define ARK_GPU_LOGGING_HPP_

#include "gpu/gpu.hpp"
#include "logging.h"

#define GLOG(cmd)                                           \
    do {                                                    \
        ark::gpuError _e = cmd;                             \
        if (_e != ark::gpuSuccess) {                        \
            const char *_estr = ark::gpuGetErrorString(_e); \
            ERR(ark::GpuError, _e, " '", _estr, "'");       \
        }                                                   \
    } while (0)

#define GLOG_DRV(cmd)                                                          \
    do {                                                                       \
        ark::gpuDrvError _e = cmd;                                             \
        if (_e != ark::gpuDrvSuccess) {                                        \
            const char *_estr;                                                 \
            if (ark::gpuDrvGetErrorString(_e, &_estr) == ark::gpuDrvSuccess) { \
                ERR(ark::GpuError, _e, " '", _estr, "'");                      \
            } else {                                                           \
                ERR(ark::GpuError, _e);                                        \
            }                                                                  \
        }                                                                      \
    } while (0)

#endif  // ARK_GPU_LOGGING_HPP_
