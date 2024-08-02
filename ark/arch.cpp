// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "arch.hpp"

#include <map>

#include "logging.hpp"

namespace ark {

Arch::Arch(const std::string &c0) : category_({c0}), name_(c0) {
    if (c0 == "ANY") category_.clear();
}

Arch::Arch(const std::string &c0, const std::string &c1)
    : category_({c0, c1}), name_(c0 + "_" + c1) {}

Arch::Arch(const std::string &c0, const std::string &c1, const std::string &c2)
    : category_({c0, c1, c2}), name_(c0 + "_" + c1 + "_" + c2) {}

Arch &Arch::operator=(const Arch &other) {
    category_ = other.category_;
    name_ = other.name_;
    return *this;
}

bool Arch::operator==(const Arch &other) const {
    if (category_.size() != other.category_.size()) return false;
    for (size_t i = 0; i < category_.size(); ++i) {
        if (category_[i] != other.category_[i]) {
            return false;
        }
    }
    return true;
}

bool Arch::belongs_to(const ArchRef arch) const {
    if (category_.size() < arch->category().size()) {
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

bool Arch::later_than(const ArchRef arch) const {
    if (category_.size() != arch->category().size()) {
        return false;
    }
    size_t idx = 0;
    for (const auto &name : arch->category()) {
        if (category_[idx] != name) {
            return category_[idx] > name;
        }
        idx++;
    }
    return false;
}

extern const ArchRef ARCH_ANY = std::make_shared<Arch>("ANY");
extern const ArchRef ARCH_CUDA = std::make_shared<Arch>("CUDA");
extern const ArchRef ARCH_CUDA_70 = std::make_shared<Arch>("CUDA", "70");
extern const ArchRef ARCH_CUDA_80 = std::make_shared<Arch>("CUDA", "80");
extern const ArchRef ARCH_CUDA_90 = std::make_shared<Arch>("CUDA", "90");
extern const ArchRef ARCH_ROCM = std::make_shared<Arch>("ROCM");
extern const ArchRef ARCH_ROCM_90A = std::make_shared<Arch>("ROCM", "90A");
extern const ArchRef ARCH_ROCM_942 = std::make_shared<Arch>("ROCM", "942");

const ArchRef Arch::from_name(const std::string &type_name) {
    static std::map<std::string, const ArchRef> instances;
    if (instances.empty()) {
        instances.emplace("ANY", ARCH_ANY);
        instances.emplace("CUDA", ARCH_CUDA);
        instances.emplace("CUDA_70", ARCH_CUDA_70);
        instances.emplace("CUDA_80", ARCH_CUDA_80);
        instances.emplace("CUDA_90", ARCH_CUDA_90);
        instances.emplace("ROCM", ARCH_ROCM);
        instances.emplace("ROCM_90A", ARCH_ROCM_90A);
        instances.emplace("ROCM_942", ARCH_ROCM_942);
    }
    auto it = instances.find(type_name);
    if (it == instances.end()) {
        ERR(InvalidUsageError, "Unknown architecture type: ", type_name);
    }
    return it->second;
}

}  // namespace ark
