// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_OP_CONFIG_HPP_
#define ARK_MODEL_OP_CONFIG_HPP_

#include <map>
#include <memory>
#include <vector>

#include "named_type.hpp"

namespace ark {

class ModelOpConfigArchT : public NamedT {
   public:
    ModelOpConfigArchT();

    ModelOpConfigArchT(const std::string &c0);

    ModelOpConfigArchT(const std::string &c0, const std::string &c1);

    ModelOpConfigArchT(const std::string &c0, const std::string &c1,
                       const std::string &c2);

    ModelOpConfigArchT(const ModelOpConfigArchT &) = default;

    const std::vector<std::string> &category() const { return category_; }

    bool belongs_to(const std::shared_ptr<ModelOpConfigArchT> arch) const;

    bool later_than(const std::shared_ptr<ModelOpConfigArchT> arch) const;

   private:
    std::vector<std::string> category_;
};

using ModelOpConfigArchType = std::shared_ptr<ModelOpConfigArchT>;

extern const ModelOpConfigArchType ARCH_ANY;

extern const ModelOpConfigArchType ARCH_CUDA;
extern const ModelOpConfigArchType ARCH_CUDA_70;
extern const ModelOpConfigArchType ARCH_CUDA_80;
extern const ModelOpConfigArchType ARCH_CUDA_90;

extern const ModelOpConfigArchType ARCH_ROCM;
extern const ModelOpConfigArchType ARCH_ROCM_90A;
extern const ModelOpConfigArchType ARCH_ROCM_942;

class ModelOpConfig {
   public:
    ModelOpConfig(const ModelOpConfigArchType arch, const std::string &name,
                  const std::string &impl_name)
        : arch_(arch), name_(name), impl_name_(impl_name) {}

    ModelOpConfig(const ModelOpConfig &) = default;

    const ModelOpConfigArchType arch() const { return arch_; }

    const std::string &name() const { return name_; }

    const std::string &impl_name() const { return impl_name_; }

   private:
    ModelOpConfigArchType arch_;
    std::string name_;
    std::string impl_name_;
};

}  // namespace ark

#endif  // ARK_MODEL_OP_CONFIG_HPP_
