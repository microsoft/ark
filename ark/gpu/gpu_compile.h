// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_COMPILE_H_
#define ARK_GPU_COMPILE_H_

#include <string>
#include <vector>

#include "ark/gpu/gpu_mgr.h"

namespace ark {

const std::string gpu_compile(const std::vector<std::string> &codes,
                              const GpuArchType &arch, unsigned int max_reg_cnt,
                              bool use_comm_sw);

} // namespace ark

#endif // ARK_GPU_COMPILE_H_
