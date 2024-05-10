// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "arch.hpp"

#include <map>

#include "logging.h"

namespace ark {

Arch::Arch(const std::string &c0) : category_({c0}), name_(c0) {
    if (c0 == "ANY") category_.clear();
}

Arch::Arch(const std::string &c0, const std::string &c1)
    : category_({c0, c1}), name_(c0 + "_" + c1) {}

Arch::Arch(const std::string &c0, const std::string &c1, const std::string &c2)
    : category_({c0, c1, c2}), name_(c0 + "_" + c1 + "_" + c2) {}

bool Arch::operator==(const Arch &other) const {
    if (category_.size() != other.category_.size()) return false;
    for (size_t i = 0; i < category_.size(); ++i) {
        if (category_[i] != other.category_[i]) {
            return false;
        }
    }
    return true;
}

bool Arch::belongs_to(const Arch &arch) const {
    if (category_.size() <= arch.category().size()) {
        return false;
    }
    size_t idx = 0;
    for (const auto &name : arch.category()) {
        if (category_[idx++] != name) {
            return false;
        }
    }
    return true;
}

bool Arch::later_than(const Arch &arch) const {
    if (category_.size() != arch.category().size()) {
        return false;
    }
    size_t idx = 0;
    for (const auto &name : arch.category()) {
        if (category_[idx] != name) {
            return category_[idx] > name;
        }
    }
    return true;
}

#define _NARGS(_1, _2, _3, _N, ...) _N
#define NARGS(...) _NARGS(__VA_ARGS__, 3, 2, 1, 0)
#define CONCAT(x, y) x##y
#define CONCAT_US(x, y) x_##y

#define _ARCH_INSTANCE_1(_c0) extern const Arch ARCH_##_c0(#_c0);
#define _ARCH_INSTANCE_2(_c0, _c1) \
    extern const Arch CONCAT_US(CONCAT_US(ARCH, _c0), _c1)(#_c0, #_c1);
#define _ARCH_INSTANCE_3(_c0, _c1, _c2) \
    extern const Arch ARCH_##_c0_##_c1_##_c2(#_c0, #_c1, #_c2);
#define _ARCH_INSTANCE(_N, ...) CONCAT(_ARCH_INSTANCE_, _N)(__VA_ARGS__)
#define ARCH_INSTANCE(...) _ARCH_INSTANCE(NARGS(__VA_ARGS__), __VA_ARGS__)

#define _ARCH_REGISTER_1(_c0) instances[#_c0] = &ARCH_##_c0
#define _ARCH_REGISTER_2(_c0, _c1) \
    instances[#_c0 "_" #_c1] = &CONCAT_US(CONCAT_US(ARCH, _c0), _c1)
#define _ARCH_REGISTER_3(_c0, _c1, _c2) \
    instances[#_c0 "_" #_c1 "_" #_c2] = &ARCH_##_c0_##_c1_##_c2
#define _ARCH_REGISTER(_N, ...) CONCAT(_ARCH_REGISTER_, _N)(__VA_ARGS__)
#define ARCH_REGISTER(...) _ARCH_REGISTER(NARGS(__VA_ARGS__), __VA_ARGS__)

// ARCH_INSTANCE(ANY);
// ARCH_INSTANCE(CUDA);
// ARCH_INSTANCE(CUDA, 70);
// ARCH_INSTANCE(CUDA, 80);
// ARCH_INSTANCE(CUDA, 90);
// ARCH_INSTANCE(ROCM);
// ARCH_INSTANCE(ROCM, 90A);
// ARCH_INSTANCE(ROCM, 942);

extern const Arch ARCH_ANY("ANY");
extern const Arch ARCH_CUDA("CUDA");
extern const Arch ARCH_CUDA_70("CUDA", "70");
extern const Arch ARCH_CUDA_80("CUDA", "80");
extern const Arch ARCH_CUDA_90("CUDA", "90");
extern const Arch ARCH_ROCM("ROCM");
extern const Arch ARCH_ROCM_90A("ROCM", "90A");
extern const Arch ARCH_ROCM_942("ROCM", "942");

const Arch &Arch::from_name(const std::string &type_name) {
    static std::map<std::string, const Arch *> instances;
    if (instances.empty()) {
        // ARCH_REGISTER(ANY);
        // ARCH_REGISTER(CUDA);
        // ARCH_REGISTER(CUDA, 70);
        // ARCH_REGISTER(CUDA, 80);
        // ARCH_REGISTER(CUDA, 90);
        // ARCH_REGISTER(ROCM);
        // ARCH_REGISTER(ROCM, 90A);
        // ARCH_REGISTER(ROCM, 942);
        instances["ANY"] = &ARCH_ANY;
        instances["CUDA"] = &ARCH_CUDA;
        instances["CUDA_70"] = &ARCH_CUDA_70;
        instances["CUDA_80"] = &ARCH_CUDA_80;
        instances["CUDA_90"] = &ARCH_CUDA_90;
        instances["ROCM"] = &ARCH_ROCM;
        instances["ROCM_90A"] = &ARCH_ROCM_90A;
        instances["ROCM_942"] = &ARCH_ROCM_942;
    }
    auto it = instances.find(type_name);
    if (it == instances.end()) {
        ERR(InvalidUsageError, "Unknown architecture type: ", type_name);
    }
    return *(it->second);
}

}  // namespace ark
