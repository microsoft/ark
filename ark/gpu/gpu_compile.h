// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_COMPILE_H_
#define ARK_GPU_COMPILE_H_

#include <string>
#include <vector>

#include "arch.hpp"

namespace ark {

const std::string gpu_compile(const std::vector<std::string> &codes,
                              const Arch &arch, unsigned int max_reg_cnt);

}  // namespace ark

#endif  // ARK_GPU_COMPILE_H_
