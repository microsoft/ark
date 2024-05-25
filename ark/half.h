// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#ifndef ARK_HALF_H_
#define ARK_HALF_H_

/// Borrowing CUTLASS's host-side half_t until we can move on to C++23 and use
/// std::float16_t

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// IEEE half-precision floating-point type
struct alignas(2) half_t {
    //
    // Data members
    //

    /// Storage type
    uint16_t storage;

    //
    // Static conversion operators
    //

    /// Constructs from an unsigned short
    static half_t bitcast(uint16_t x) {
        half_t h;
        h.storage = x;
        return h;
    }

    /// FP32 -> FP16 conversion - rounds to nearest even
    static half_t convert(float const& flt) {
        // software implementation rounds toward nearest even
        unsigned s;

        std::memcpy(&s, &flt, sizeof(s));

        uint16_t sign = uint16_t((s >> 16) & 0x8000);
        int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
        int mantissa = s & 0x7fffff;
        uint16_t u = 0;

        if ((s & 0x7fffffff) == 0) {
            // sign-preserving zero
            return bitcast(sign);
        }

        if (exp > 15) {
            if (exp == 128 && mantissa) {
                // not a number
                u = 0x7fff;
            } else {
                // overflow to infinity
                u = sign | 0x7c00;
            }
            return bitcast(u);
        }

        int sticky_bit = 0;

        if (exp >= -14) {
            // normal fp32 to normal fp16
            exp = uint16_t(exp + uint16_t(15));
            u = uint16_t(((exp & 0x1f) << 10));
            u = uint16_t(u | (mantissa >> 13));
        } else {
            // normal single-precision to subnormal half_t-precision
            // representation
            int rshift = (-14 - exp);
            if (rshift < 32) {
                mantissa |= (1 << 23);

                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

                mantissa = (mantissa >> rshift);
                u = (uint16_t(mantissa >> 13) & 0x3ff);
            } else {
                mantissa = 0;
                u = 0;
            }
        }

        // round to nearest even
        int round_bit = ((mantissa >> 12) & 1);
        sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

        if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
            u = uint16_t(u + 1);
        }

        u |= sign;

        return bitcast(u);
    }

    /// FP32 -> FP16 conversion - rounds to nearest even
    static half_t convert(int const& n) { return convert(float(n)); }

    /// FP32 -> FP16 conversion - rounds to nearest even
    static half_t convert(unsigned const& n) { return convert(float(n)); }

    /// Converts a half-precision value stored as a uint16_t to a float
    static float convert(half_t const& x) {
        uint16_t const& h = x.storage;
        int sign = ((h >> 15) & 1);
        int exp = ((h >> 10) & 0x1f);
        int mantissa = (h & 0x3ff);
        unsigned f = 0;

        if (exp > 0 && exp < 31) {
            // normal
            exp += 112;
            f = (sign << 31) | (exp << 23) | (mantissa << 13);
        } else if (exp == 0) {
            if (mantissa) {
                // subnormal
                exp += 113;
                while ((mantissa & (1 << 10)) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x3ff;
                f = (sign << 31) | (exp << 23) | (mantissa << 13);
            } else {
                // sign-preserving zero
                f = (sign << 31);
            }
        } else if (exp == 31) {
            if (mantissa) {
                f = 0x7fffffff;  // not a number
            } else {
                f = (0xff << 23) | (sign << 31);  //  inf
            }
        }
        float flt;
        std::memcpy(&flt, &f, sizeof(flt));
        return flt;
    }

    //
    // Methods
    //

    /// Default constructor
    half_t() = default;

    /// Floating point conversion
    half_t(float x) { storage = convert(x).storage; }

    /// Floating point conversion
    half_t(double x) : half_t(float(x)) {}

    /// Integer conversion - round to nearest even
    half_t(int x) { storage = convert(x).storage; }

    /// Integer conversion - round toward zero
    half_t(unsigned x) { storage = convert(x).storage; }

    /// Converts to float
    operator float() const { return convert(*this); }

    /// Converts to double
    explicit operator double() const { return double(convert(*this)); }

    /// Converts to int
    explicit operator int() const { return int(convert(*this)); }

    /// Casts to bool
    explicit operator bool() const { return (convert(*this) != 0.0f); }

    /// Assignment
    template <typename T>
    half_t& operator=(T const& x) {
        storage = convert(float(x)).storage;
        return *this;
    }

    /// Accesses raw internal state
    uint16_t& raw() { return storage; }

    /// Accesses raw internal state
    uint16_t raw() const { return storage; }

    /// Returns the sign bit
    bool signbit() const { return ((storage & 0x8000) != 0); }

    /// Returns the biased exponent
    int exponent_biased() const { return int((storage >> 10) & 0x1f); }

    /// Returns the unbiased exponent
    int exponent() const { return exponent_biased() - 15; }

    /// Returns the mantissa
    int mantissa() const { return int(storage & 0x3ff); }
};

using fp16 = half_t;

/// Assignment from half_t
template <>
half_t& half_t::operator=(half_t const& x);

/// Assignment from float
template <>
half_t& half_t::operator=(float const& x);

///////////////////////////////////////////////////////////////////////////////////////////////////

bool signbit(ark::half_t const& h);

ark::half_t abs(ark::half_t const& h);

bool isnan(ark::half_t const& h);

bool isfinite(ark::half_t const& h);

ark::half_t nanh(const char*);

bool isinf(ark::half_t const& h);

bool isnormal(ark::half_t const& h);

int fpclassify(ark::half_t const& h);

ark::half_t sqrt(ark::half_t const& h);

half_t copysign(half_t const& a, half_t const& b);

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace std {

/// Numeric limits
template <>
struct numeric_limits<ark::half_t> {
    static bool const is_specialized = true;
    static bool const is_signed = true;
    static bool const is_integer = false;
    static bool const is_exact = false;
    static bool const has_infinity = true;
    static bool const has_quiet_NaN = true;
    static bool const has_signaling_NaN = false;
    static std::float_denorm_style const has_denorm = std::denorm_present;
    static bool const has_denorm_loss = true;
    static std::float_round_style const round_style = std::round_to_nearest;
    static bool const is_iec559 = true;
    static bool const is_bounded = true;
    static bool const is_modulo = false;
    static int const digits = 10;

    /// Least positive value
    static ark::half_t min() { return ark::half_t::bitcast(0x0001); }

    /// Minimum finite value
    static ark::half_t lowest() { return ark::half_t::bitcast(0xfbff); }

    /// Maximum finite value
    static ark::half_t max() { return ark::half_t::bitcast(0x7bff); }

    /// Returns smallest finite value
    static ark::half_t epsilon() { return ark::half_t::bitcast(0x1800); }

    /// Returns maximum rounding error
    static ark::half_t round_error() { return ark::half_t(0.5f); }

    /// Returns positive infinity value
    static ark::half_t infinity() { return ark::half_t::bitcast(0x7c00); }

    /// Returns quiet NaN value
    static ark::half_t quiet_NaN() { return ark::half_t::bitcast(0x7fff); }

    /// Returns signaling NaN value
    static ark::half_t signaling_NaN() { return ark::half_t::bitcast(0x7fff); }

    /// Returns smallest positive subnormal value
    static ark::half_t denorm_min() { return ark::half_t::bitcast(0x0001); }
};

ark::half_t abs(ark::half_t const& h);

ark::half_t max(ark::half_t const& a, ark::half_t const& b);

ark::half_t exp(ark::half_t const& h);

}  // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

bool operator==(half_t const& lhs, half_t const& rhs);

bool operator!=(half_t const& lhs, half_t const& rhs);

bool operator<(half_t const& lhs, half_t const& rhs);

bool operator<=(half_t const& lhs, half_t const& rhs);

bool operator>(half_t const& lhs, half_t const& rhs);

bool operator>=(half_t const& lhs, half_t const& rhs);

half_t operator+(half_t const& lhs, half_t const& rhs);

half_t operator-(half_t const& lhs);

half_t operator-(half_t const& lhs, half_t const& rhs);

half_t operator*(half_t const& lhs, half_t const& rhs);

half_t operator/(half_t const& lhs, half_t const& rhs);

half_t& operator+=(half_t& lhs, half_t const& rhs);

half_t& operator-=(half_t& lhs, half_t const& rhs);

half_t& operator*=(half_t& lhs, half_t const& rhs);

half_t& operator/=(half_t& lhs, half_t const& rhs);

half_t& operator++(half_t& lhs);

half_t& operator--(half_t& lhs);

half_t operator++(half_t& lhs, int);

half_t operator--(half_t& lhs, int);

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

ark::half_t operator"" _hf(long double x);

ark::half_t operator"" _hf(unsigned long long int x);

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // ARK_HALF_H_
