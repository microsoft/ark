// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_LOGGING_H_
#define ARK_GPU_LOGGING_H_

#include "gpu/gpu.h"
#include "include/ark.h"
#include "logging.h"

#define GLOG(cmd)                                                        \
    do {                                                                 \
        ark::gpuError _e = cmd;                                          \
        if (_e != ark::gpuSuccess) {                                     \
            const char *_estr;                                           \
            if (ark::gpuGetErrorString(_e, &_estr) == ark::gpuSuccess) { \
                ERR(ark::GpuError, _e, " '", _estr, "'");                \
            } else {                                                     \
                ERR(ark::GpuError, _e);                                  \
            }                                                            \
        }                                                                \
    } while (0)

#endif  // ARK_GPU_LOGGING_H_
