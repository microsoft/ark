// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"

// clang-format off
#include "vector_types.h"
#include "cutlass/bfloat16.h"
// clang-format on

/// Convert cutlass::bfloat16_t to @ref ark::bfloat16_t
/// @param cub cutlass::bfloat16_t
/// @return @ref ark::bfloat16_t
inline static const ark::bfloat16_t convert(const cutlass::bfloat16_t &cub) {
    ark::bfloat16_t ret;
    ret.storage = cub.raw();
    return ret;
}

/// Numeric limits of @ref ark::bfloat16_t
template <>
struct std::numeric_limits<ark::bfloat16_t> {
    static ark::bfloat16_t max() {
        return convert(std::numeric_limits<cutlass::bfloat16_t>::max());
    }
    static ark::bfloat16_t min() {
        return convert(std::numeric_limits<cutlass::bfloat16_t>::min());
    }
    static ark::bfloat16_t epsilon() {
        return convert(std::numeric_limits<cutlass::bfloat16_t>::epsilon());
    }
};

ark::bfloat16_t operator+(ark::bfloat16_t const &lhs,
                          ark::bfloat16_t const &rhs) {
    return convert(cutlass::bfloat16_t::bitcast(lhs.storage) +
                   cutlass::bfloat16_t::bitcast(rhs.storage));
}

ark::bfloat16_t operator-(ark::bfloat16_t const &lhs,
                          ark::bfloat16_t const &rhs) {
    return convert(cutlass::bfloat16_t::bitcast(lhs.storage) -
                   cutlass::bfloat16_t::bitcast(rhs.storage));
}

ark::bfloat16_t operator*(ark::bfloat16_t const &lhs,
                          ark::bfloat16_t const &rhs) {
    return convert(cutlass::bfloat16_t::bitcast(lhs.storage) *
                   cutlass::bfloat16_t::bitcast(rhs.storage));
}

ark::bfloat16_t &operator+=(ark::bfloat16_t &lhs, ark::bfloat16_t const &rhs) {
    cutlass::bfloat16_t v = cutlass::bfloat16_t::bitcast(lhs.storage) +
                            cutlass::bfloat16_t::bitcast(rhs.storage);
    lhs.storage = v.raw();
    return lhs;
}

ark::bfloat16_t &operator-=(ark::bfloat16_t &lhs, ark::bfloat16_t const &rhs) {
    cutlass::bfloat16_t v = cutlass::bfloat16_t::bitcast(lhs.storage) -
                            cutlass::bfloat16_t::bitcast(rhs.storage);
    lhs.storage = v.raw();
    return lhs;
}

/// Return the absolute value of a @ref ark::bfloat16_t
/// @param val Input value
/// @return @ref Absolute value of `val`
ark::bfloat16_t abs(ark::bfloat16_t const &val) {
    return convert(cutlass::abs(cutlass::bfloat16_t::bitcast(val.storage)));
}

namespace ark {

/// Construct a @ref bfloat16_t from a float
/// @param f Input value
bfloat16_t::bfloat16_t(float f) {
    this->storage = cutlass::bfloat16_t(f).raw();
}

/// Convert a @ref bfloat16_t to a float
/// @return float
bfloat16_t::operator float() const {
    return float(cutlass::bfloat16_t::bitcast(this->storage));
}

}  // namespace ark
