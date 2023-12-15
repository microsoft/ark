// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_INTEGER_H_
#define ARK_KERNELS_INTEGER_H_

#include "device.h"
#include "vector_type.h"

namespace ark {

using i32 = int;
using i32x2 = int2;
using i32x4 = int4;
using ui32 = unsigned int;
using ui32x2 = uint2;
using ui32x4 = uint4;

namespace type {

template <typename DataType>
struct Constant;

template <>
struct Constant<i32> {
    static DEVICE i32 zero() { return 0; }
    static DEVICE i32 lowest() { return 0x80000000; }
};

template <>
struct Constant<i32x2> {
    static DEVICE i32x2 zero() { return make_int2(0, 0); }
    static DEVICE i32x2 lowest() { return make_int2(0x80000000, 0x80000000); }
};

template <>
struct Constant<i32x4> {
    static DEVICE i32x4 zero() { return make_int4(0, 0, 0, 0); }
    static DEVICE i32x4 lowest() {
        return make_int4(0x80000000, 0x80000000, 0x80000000, 0x80000000);
    }
};

template <>
struct Constant<ui32> {
    static DEVICE ui32 zero() { return 0; }
    static DEVICE ui32 lowest() { return 0; }
};

template <>
struct Constant<ui32x2> {
    static DEVICE ui32x2 zero() { return make_uint2(0, 0); }
    static DEVICE ui32x2 lowest() { return make_uint2(0, 0); }
};

template <>
struct Constant<ui32x4> {
    static DEVICE ui32x4 zero() { return make_uint4(0, 0); }
    static DEVICE ui32x4 lowest() { return make_uint4(0, 0); }
};

template <>
struct Vtype<i32, 2> {
    using type = i32x2;
};

template <>
struct Vtype<i32, 4> {
    using type = i32x4;
};

template <>
struct Vtype<ui32, 2> {
    using type = ui32x2;
};

template <>
struct Vtype<ui32, 4> {
    using type = ui32x4;
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_INTEGER_H_
