// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_CHECKER_H_
#define ARK_KERNELS_CHECKER_H_

#include "device.h"

namespace ark {

/// Check if two values are equal at compile time.
/// @tparam Value0_ First value.
/// @tparam Value1_ Second value.
template <int Value0_, int Value1_> struct IsEq
{
    static const int Value0 = Value0_;
    static const int Value1 = Value1_;
    static_assert(Value0 == Value1, "Size mismatch");
    DEVICE void operator()() const
    {
        // Do nothing.
    }
};

} // namespace ark

#endif // ARK_KERNELS_CHECKER_H_
