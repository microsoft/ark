// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_op_config.hpp"

namespace ark {

ModelOpConfigArchT::ModelOpConfigArchT() : ModelNamedT("ANY"){};

ModelOpConfigArchT::ModelOpConfigArchT(const std::string &c0)
    : ModelNamedT(c0), category_({c0}) {}

ModelOpConfigArchT::ModelOpConfigArchT(const std::string &c0,
                                       const std::string &c1)
    : ModelNamedT(c0 + "_" + c1), category_({c0, c1}) {}

ModelOpConfigArchT::ModelOpConfigArchT(const std::string &c0,
                                       const std::string &c1,
                                       const std::string &c2)
    : ModelNamedT(c0 + "_" + c1 + "_" + c2), category_({c0, c1, c2}) {}

bool ModelOpConfigArchT::belongs_to(
    const std::shared_ptr<ModelOpConfigArchT> arch) const {
    if (category_.size() <= arch->category().size()) {
        return false;
    }
    size_t idx = 0;
    for (const auto &name : arch->category()) {
        if (category_[idx++] != name) {
            return false;
        }
    }
    return true;
}

bool ModelOpConfigArchT::later_than(
    const std::shared_ptr<ModelOpConfigArchT> arch) const {
    if (category_.size() != arch->category().size()) {
        return false;
    }
    size_t idx = 0;
    for (const auto &name : arch->category()) {
        if (category_[idx] != name) {
            return category_[idx] > name;
        }
    }
    return true;
}

extern const ModelOpConfigArchType ARCH_ANY =
    std::make_shared<ModelOpConfigArchT>();

extern const ModelOpConfigArchType ARCH_CUDA =
    std::make_shared<ModelOpConfigArchT>("CUDA");
extern const ModelOpConfigArchType ARCH_CUDA_70 =
    std::make_shared<ModelOpConfigArchT>("CUDA", "70");
extern const ModelOpConfigArchType ARCH_CUDA_80 =
    std::make_shared<ModelOpConfigArchT>("CUDA", "80");
extern const ModelOpConfigArchType ARCH_CUDA_90 =
    std::make_shared<ModelOpConfigArchT>("CUDA", "90");

extern const ModelOpConfigArchType ARCH_ROCM =
    std::make_shared<ModelOpConfigArchT>("ROCM");
extern const ModelOpConfigArchType ARCH_ROCM_90A =
    std::make_shared<ModelOpConfigArchT>("ROCM", "90A");
extern const ModelOpConfigArchType ARCH_ROCM_942 =
    std::make_shared<ModelOpConfigArchT>("ROCM", "942");

}  // namespace ark
