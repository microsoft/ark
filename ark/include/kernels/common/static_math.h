// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_STATIC_MATH_H_
#define ARK_KERNELS_STATIC_MATH_H_

#include "device.h"

namespace ark {
namespace math {

// Absolute value
template <long long int N>
struct abs {
    enum { value = (N > 0) ? N : -N };
};

// Larger value
template <long long int A, long long int B>
struct max {
    enum { value = (A > B) ? A : B };
};

// Smaller value
template <long long int A, long long int B>
struct min {
    enum { value = (A < B) ? A : B };
};

// Statically determine log2(N), rounded up
template <long long int N, long long int CurrentVal = N,
          long long int Count = 0>
struct log2_up {
    enum { value = log2_up<N, (CurrentVal >> 1), Count + 1>::value };
};

template <long long int N, long long int Count>
struct log2_up<N, 1, Count> {
    static_assert(N > 0, "invalid input domain.");
    enum { value = ((1 << Count) < N) ? Count + 1 : Count };
};

// Statically determine log2(N), rounded down
template <long long int N, long long int CurrentVal = N,
          long long int Count = 0>
struct log2_down {
    enum { value = log2_down<N, (CurrentVal >> 1), Count + 1>::value };
};

template <long long int N, long long int Count>
struct log2_down<N, 1, Count> {
    static_assert(N > 0, "invalid input domain.");
    enum { value = Count };
};

////////////////////////////////////////////////////////////////////////////////

// Safe multiplication for preventing overflow.
template <long long int A, long long int B>
struct mul {
    enum { value = A * B };
    static_assert(value / A == B, "overflow detected.");
};

////////////////////////////////////////////////////////////////////////////////

// Integer division, rounded up
template <long long int A, long long int B>
struct div_up {
    enum { value = (A + B - 1) / B };
};

// Least multiple of B equal to or larger than A
template <long long int A, long long int B>
struct lm {
    enum { value = mul<div_up<A, B>::value, B>::value };
};

// Integer subtraction
template <long long int A, long long int B>
struct sub {
    enum { value = A - B };
};

// 1 if N is power of 2, otherwise 0
template <long long int N>
struct is_pow2 {
    enum { value = N && (!(N & (N - 1))) };
};

//
template <long long int X, long long int Pad>
struct pad {
    enum { value = mul<div_up<X, Pad>::value, Pad>::value };
};

////////////////////////////////////////////////////////////////////////////////

// Helper of div.
template <long long int Divisor, bool IsPow2>
struct Div {};

template <long long int Divisor>
struct Div<Divisor, true> {
    static DEVICE long long int compute(long long int x) {
        return x >> math::log2_up<Divisor>::value;
    }
};

template <long long int Divisor>
struct Div<Divisor, false> {
    static DEVICE long long int compute(long long int x) { return x / Divisor; }
};

// Fast division by pow2 divisor.
template <long long int Divisor>
static DEVICE long long int div(long long int x) {
    return Div<Divisor, math::is_pow2<Divisor>::value>::compute(x);
}

////////////////////////////////////////////////////////////////////////////////

// Helper of mod.
template <long long int Divisor, bool IsPow2>
struct Mod {};

template <long long int Divisor>
struct Mod<Divisor, true> {
    static DEVICE long long int compute(long long int x) {
        return x & (Divisor - 1);
    }
};

template <long long int Divisor>
struct Mod<Divisor, false> {
    static DEVICE long long int compute(long long int x) { return x % Divisor; }
};

// Fast modulo by pow2 divisor.
template <long long int Divisor>
static DEVICE long long int mod(long long int x) {
    return Mod<Divisor, math::is_pow2<Divisor>::value>::compute(x);
}

////////////////////////////////////////////////////////////////////////////////

/// Greatest multiple of Divisor equal to or smaller than x
template <long long int Divisor>
static DEVICE long long int gm(long long int x) {
    return math::div<Divisor>(x) * Divisor;
}

////////////////////////////////////////////////////////////////////////////////

template <size_t Rhs>
DEVICE bool geq(size_t x) {
    return x >= Rhs;
}

template <>
DEVICE bool geq<0>(size_t x) {
    return true;
}

template <size_t Rhs>
DEVICE bool le(size_t x) {
    return x < Rhs;
}

template <>
DEVICE bool le<0>(size_t x) {
    return false;
}

}  // namespace math
}  // namespace ark

#endif  // ARK_KERNELS_STATIC_MATH_H_
