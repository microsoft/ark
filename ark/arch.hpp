// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_ARCH_HPP_
#define ARK_ARCH_HPP_

#include <memory>
#include <string>
#include <vector>

namespace ark {

class Arch;
using ArchRef = std::shared_ptr<Arch>;

class Arch {
   private:
    std::vector<std::string> category_;
    std::string name_;

   public:
    Arch(){};

    Arch(const std::string &c0);

    Arch(const std::string &c0, const std::string &c1);

    Arch(const std::string &c0, const std::string &c1, const std::string &c2);

    Arch(const Arch &other) = default;

    Arch &operator=(const Arch &other);

    const std::string &name() const { return name_; }

    bool operator==(const Arch &other) const;

    bool operator!=(const Arch &other) const { return !(*this == other); }

    const std::vector<std::string> &category() const { return category_; }

    bool belongs_to(const ArchRef arch) const;

    bool later_than(const ArchRef arch) const;

    static const ArchRef from_name(const std::string &name);
};

extern const ArchRef ARCH_ANY;

extern const ArchRef ARCH_CUDA;
extern const ArchRef ARCH_CUDA_70;
extern const ArchRef ARCH_CUDA_80;
extern const ArchRef ARCH_CUDA_90;

extern const ArchRef ARCH_ROCM;
extern const ArchRef ARCH_ROCM_90A;
extern const ArchRef ARCH_ROCM_942;

}  // namespace ark

#endif  // ARK_ARCH_HPP_
