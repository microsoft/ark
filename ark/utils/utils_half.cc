// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"

// clang-format off
#include "vector_types.h"
#include "cutlass/half.h"
// clang-format on

/// Convert cutlass::half_t to @ref ark::half_t
/// @param cuh cutlass::half_t
/// @return @ref ark::half_t
inline static const ark::half_t convert(const cutlass::half_t &cuh) {
    ark::half_t ret;
    ret.storage = cuh.raw();
    return ret;
}

/// Numeric limits of @ref ark::half_t
template <>
struct std::numeric_limits<ark::half_t> {
    static ark::half_t max() {
        return convert(std::numeric_limits<cutlass::half_t>::max());
    }
    static ark::half_t min() {
        return convert(std::numeric_limits<cutlass::half_t>::min());
    }
    static ark::half_t epsilon() {
        return convert(std::numeric_limits<cutlass::half_t>::epsilon());
    }
};

ark::half_t operator+(ark::half_t const &lhs, ark::half_t const &rhs) {
    return convert(cutlass::half_t::bitcast(lhs.storage) +
                   cutlass::half_t::bitcast(rhs.storage));
}

ark::half_t operator-(ark::half_t const &lhs, ark::half_t const &rhs) {
    return convert(cutlass::half_t::bitcast(lhs.storage) -
                   cutlass::half_t::bitcast(rhs.storage));
}

ark::half_t operator*(ark::half_t const &lhs, ark::half_t const &rhs) {
    return convert(cutlass::half_t::bitcast(lhs.storage) *
                   cutlass::half_t::bitcast(rhs.storage));
}

ark::half_t &operator+=(ark::half_t &lhs, ark::half_t const &rhs) {
    cutlass::half_t v = cutlass::half_t::bitcast(lhs.storage) +
                        cutlass::half_t::bitcast(rhs.storage);
    lhs.storage = v.raw();
    return lhs;
}

ark::half_t &operator-=(ark::half_t &lhs, ark::half_t const &rhs) {
    cutlass::half_t v = cutlass::half_t::bitcast(lhs.storage) -
                        cutlass::half_t::bitcast(rhs.storage);
    lhs.storage = v.raw();
    return lhs;
}

/// Return the absolute value of a @ref ark::half_t
/// @param val Input value
/// @return @ref Absolute value of `val`
ark::half_t abs(ark::half_t const &val) {
    return convert(cutlass::abs(cutlass::half_t::bitcast(val.storage)));
}

namespace ark {

/// Construct a @ref half_t from a float
/// @param f Input value
half_t::half_t(float f) { this->storage = cutlass::half_t(f).raw(); }

/// Convert a @ref half_t to a float
/// @return float
half_t::operator float() const {
    return float(cutlass::half_t::bitcast(this->storage));
}

}  // namespace ark
