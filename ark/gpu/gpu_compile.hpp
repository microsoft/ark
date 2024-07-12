// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_COMPILE_HPP_
#define ARK_GPU_COMPILE_HPP_

#include <string>
#include <vector>

#include "arch.hpp"

namespace ark {

const std::string gpu_compile(const std::vector<std::string> &codes,
                              const ArchRef arch, unsigned int max_reg_cnt);

}  // namespace ark

#endif  // ARK_GPU_COMPILE_HPP_
