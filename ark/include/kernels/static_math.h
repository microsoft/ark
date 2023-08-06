// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_STATIC_MATH_H_
#define ARK_KERNELS_STATIC_MATH_H_

#include "device.h"

namespace ark {
namespace math {

// Absolute value
template <int N> struct abs
{
    enum
    {
        value = (N > 0) ? N : -N
    };
};

// Larger value
template <int A, int B> struct max
{
    enum
    {
        value = (A > B) ? A : B
    };
};

// Smaller value
template <int A, int B> struct min
{
    enum
    {
        value = (A < B) ? A : B
    };
};

// Statically determine log2(N), rounded up
template <int N, int CurrentVal = N, int Count = 0> struct log2_up
{
    enum
    {
        value = log2_up<N, (CurrentVal >> 1), Count + 1>::value
    };
};
template <int N, int Count> struct log2_up<N, 1, Count>
{
    static_assert(N > 0, "invalid input domain.");
    enum
    {
        value = ((1 << Count) < N) ? Count + 1 : Count
    };
};

////////////////////////////////////////////////////////////////////////////////

// Safe multiplication for preventing overflow.
template <int A, int B> struct mul
{
    static const int Log2AbsA = log2_up<abs<A>::value>::value;
    static const int Log2AbsB = log2_up<abs<B>::value>::value;
    static_assert(Log2AbsA + Log2AbsB <= 31, "overflow detected");
    enum
    {
        value = A * B
    };
};

////////////////////////////////////////////////////////////////////////////////

// Integer division, rounded up
template <int A, int B> struct div_up
{
    enum
    {
        value = (A + B - 1) / B
    };
};

// Least multiple of B equal to or larger than A
template <int A, int B> struct lm
{
    enum
    {
        value = mul<div_up<A, B>::value, B>::value
    };
};

// Integer subtraction
template <int A, int B> struct sub
{
    enum
    {
        value = A - B
    };
};

// 1 if N is power of 2, otherwise 0
template <int N> struct is_pow2
{
    enum
    {
        value = N && (!(N & (N - 1)))
    };
};

//
template <int X, int Pad> struct pad
{
    enum
    {
        value = mul<div_up<X, Pad>::value, Pad>::value
    };
};

////////////////////////////////////////////////////////////////////////////////

// Helper of div.
template <int Divisor, bool IsPow2> struct Div
{
};

template <int Divisor> struct Div<Divisor, true>
{
    static DEVICE int compute(int x)
    {
        return x >> math::log2_up<Divisor>::value;
    }
};

template <int Divisor> struct Div<Divisor, false>
{
    static DEVICE int compute(int x)
    {
        return x / Divisor;
    }
};

// Fast division by pow2 divisor.
template <int Divisor> static DEVICE int div(int x)
{
    return Div<Divisor, math::is_pow2<Divisor>::value>::compute(x);
}

////////////////////////////////////////////////////////////////////////////////

// Helper of mod.
template <int Divisor, bool IsPow2> struct Mod
{
};

template <int Divisor> struct Mod<Divisor, true>
{
    static DEVICE int compute(int x)
    {
        return x & (Divisor - 1);
    }
};

template <int Divisor> struct Mod<Divisor, false>
{
    static DEVICE int compute(int x)
    {
        return x % Divisor;
    }
};

// Fast modulo by pow2 divisor.
template <int Divisor> static DEVICE int mod(int x)
{
    return Mod<Divisor, math::is_pow2<Divisor>::value>::compute(x);
}

////////////////////////////////////////////////////////////////////////////////

/// Greatest multiple of Divisor equal to or smaller than x
template <int Divisor> static DEVICE int gm(int x)
{
    return math::div<Divisor>(x) * Divisor;
}

} // namespace math
} // namespace ark

#endif // ARK_KERNELS_STATIC_MATH_H_
