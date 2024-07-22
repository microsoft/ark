// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#ifndef ARK_FP8_H_
#define ARK_FP8_H_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  FP8 Has 2 encodings possible : E4M3 and E5M2
//
//  E4M3 : 7  |  6 5 4 3  |  2 1 0
//  E5M2 : 7  |  6 5 4 3 2  |  1 0
//
///////////////////////////////////////////////////////////////////////////////////////////////////

enum class FloatEncoding { E4M3, E5M2 };

template <FloatEncoding T>
struct alignas(1) float8_base {
    static constexpr bool IS_E4M3 = (T == FloatEncoding::E4M3);
    static constexpr bool IS_E5M2 = (T == FloatEncoding::E5M2);

    // Number of Bits representing mantissa and exponents
    static constexpr int FP32_NUM_BITS = 32;
    static constexpr int FP32_NUM_EXPONENT_BITS = 8;
    static constexpr int FP32_NUM_MANTISSA_BITS = 23;
    static constexpr uint32_t FP32_NAN = 0x7fffffff;
    static constexpr uint32_t FP32_INFINITY_MASK = 0x7f800000;
    static constexpr int FP32_MAX_EXPONENT = 127;
    static constexpr int FP32_MIN_EXPONENT = -126;
    static constexpr int FP32_EXPONENT_BIAS = 127;

    static constexpr int FP16_NUM_BITS = 16;
    static constexpr int FP16_NUM_EXPONENT_BITS = 5;
    static constexpr int FP16_NUM_MANTISSA_BITS = 10;
    static constexpr uint16_t FP16_NAN = 0x7fff;
    static constexpr uint16_t FP16_INFINITY_MASK = 0x7c00;
    static constexpr int FP16_MAX_EXPONENT = 15;
    static constexpr int FP16_MIN_EXPONENT = -14;
    static constexpr int FP16_EXPONENT_BIAS = 15;

    static constexpr int FP8_NUM_BITS = 8;
    static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
    static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
    static constexpr uint8_t FP8_NAN = 0x7f;  // Also F8_INF
    static constexpr uint8_t FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;
    static constexpr int FP8_MAX_EXPONENT = IS_E4M3 ? 7 : 15;
    static constexpr int FP8_MIN_EXPONENT = IS_E4M3 ? -6 : -14;
    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ? 7 : 15;

    static constexpr uint8_t FP8_EXPONENT_MASK =
        (1 << FP8_NUM_EXPONENT_BITS) - 1;
    static constexpr uint8_t FP8_MANTISSA_MASK =
        (1 << FP8_NUM_MANTISSA_BITS) - 1;

    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);

    // 256 in float
    static constexpr uint32_t FP8_SAT_VAL_FP32 = 0x43800000;

    //
    // Data members
    //

    /// Data container
    uint8_t storage;

    /// Ctors.
    float8_base() : storage(0) {}

    /// Is finite implementation
    static bool isfinite(float flt) {
        uint32_t s;
        std::memcpy(&s, &flt, sizeof(s));
        return (s & 0x7f800000) < 0x7f800000;
    }

    /// Is NaN implementation
    static bool isnan(float flt) {
        uint32_t s;
        std::memcpy(&s, &flt, sizeof(s));
        return (s & 0x7fffffff) > 0x7f800000;
    }

    /// Is infinite implementation
    static bool isinf(float flt) {
        uint32_t s;
        std::memcpy(&s, &flt, sizeof(s));
        // Sign = 0 for +inf, 1 for -inf
        // Exponent = all ones
        // Mantissa = all zeros
        return (s == 0x7f800000) || (s == 0xff800000);
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    static uint8_t convert_float_to_fp8(float const& flt) {
        // software implementation rounds toward nearest even
        uint32_t s;

        std::memcpy(&s, &flt, sizeof(s));

        // Extract the bits in the FP32 type
        uint8_t sign = uint8_t((s >> 24 & 0x80));
        int32_t exp =
            int32_t((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS;
        int mantissa = s & 0x7fffff;
        uint8_t u = 0;

        uint8_t const kF8_NaN = 0x7f;

        // NaN => NaN
        if (isnan(flt)) {
            return kF8_NaN;
        }

        // Inf => MAX_FLT (satfinite)
        if (isinf(flt)) {
            return sign | FP8_MAX_FLT;
        }

        // Special handling
        if (exp == -128) {
            // int8 range is from -128 to 127
            // So 255(inf) - 127(bias) = 128 - will show up as -128

            // satfinite
            return (sign | FP8_MAX_FLT);
        }

        int sticky_bit = 0;

        bool skip_sign = false;
        bool may_be_nan = false;

        if ((exp >= FP8_MIN_EXPONENT) && (exp <= FP8_MAX_EXPONENT)) {
            // normal fp32 to normal fp8
            exp = exp + FP8_EXPONENT_BIAS;
            u = uint8_t((uint32_t(exp) & FP8_EXPONENT_MASK)
                        << FP8_NUM_MANTISSA_BITS);
            u = uint8_t(u | (mantissa >>
                             (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
        } else if (exp < FP8_MIN_EXPONENT) {
            // normal single-precision to subnormal float8-precision
            // representation
            int rshift = (FP8_MIN_EXPONENT - exp);
            if (rshift < FP32_NUM_BITS) {
                mantissa |= (1 << FP32_NUM_MANTISSA_BITS);

                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

                mantissa = (mantissa >> rshift);
                u = (uint8_t(mantissa >>
                             (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) &
                     FP8_MANTISSA_MASK);
            } else {
                mantissa = 0;
                u = 0;
            }
            // Exponent > FP8_MAX_EXPONENT - this is a special case done to
            // match HW 0x4380_0000 to 0x43e0_0000 - maps from 256 to 448, and
            // does not saturate / inf.
        } else {
            if (exp == (FP8_MAX_EXPONENT + 1)) {
                uint8_t mantissa_tmp =
                    uint8_t(mantissa >>
                            (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
                if (mantissa_tmp < FP8_MANTISSA_MASK) {
                    exp = exp + FP8_EXPONENT_BIAS;
                    u = uint8_t(uint32_t(exp) << FP8_NUM_MANTISSA_BITS) |
                        mantissa_tmp;
                    may_be_nan = (mantissa_tmp == (FP8_MANTISSA_MASK - 1));
                } else {
                    // satfinite
                    return (sign | FP8_MAX_FLT);
                }
            } else {
                // satfinite
                return (sign | FP8_MAX_FLT);
            }
        }

        // round to nearest even
        int NUM_BITS_SHIFT =
            FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
        int round_bit = ((mantissa >> NUM_BITS_SHIFT) & 1);
        sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0);

        if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
            u = uint8_t(u + 1);
            if (may_be_nan) {
                skip_sign = true;
            }
        }

        if (u > FP8_MAX_FLT) {
            // satfinite
            u = (sign | FP8_MAX_FLT);
        }

        if (!skip_sign) {
            u |= sign;
        }

        return u;
    }

    /// Converts a fp8 value stored as a uint8_t to a float
    static float convert_fp8_to_float(uint8_t const& x) {
        uint32_t constexpr kF32_NaN = 0x7fffffff;

        uint8_t const& f8 = x;
        uint32_t sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;
        uint32_t exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
        uint32_t mantissa = f8 & FP8_MANTISSA_MASK;
        unsigned f = (sign << (FP32_NUM_BITS - 1));

        if (IS_E4M3 && exp == 15 && mantissa == 0x7) {
            f = kF32_NaN;
        } else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT +
                                                 FP8_EXPONENT_BIAS + 1))) {
            // normal
            exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS);
            f = f | (exp << FP32_NUM_MANTISSA_BITS) |
                (mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
        } else if (exp == 0) {
            if (mantissa) {
                // subnormal
                exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1;
                while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= FP8_MANTISSA_MASK;
                f = f | (exp << FP32_NUM_MANTISSA_BITS) |
                    (mantissa
                     << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
            } else {
                // sign-preserving zero
            }
        } else {
            if (mantissa == 0) {
                // Sign-preserving infinity
                f = (f | 0x7f800000);
            } else {
                // Canonical NaN
                f = kF32_NaN;
            }
        }
        float flt;
        std::memcpy(&flt, &f, sizeof(flt));
        return flt;
    }
};

// Forward declaration of float_e5m2_t to define float_e4m3_t <=> float_e5m2_t
// conversions in class float_e4m3_t
struct float_e5m2_t;

///////////////////////////////////////////////////////////////
///
/// floating-point 8 type : E4M3
///
///////////////////////////////////////////////////////////////
struct alignas(1) float_e4m3_t : float8_base<FloatEncoding::E4M3> {
    using Base = float8_base<FloatEncoding::E4M3>;
    using Base::Base;

    static float_e4m3_t bitcast(uint8_t x) {
        float_e4m3_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    static float_e4m3_t from_float(float const& flt) {
        return bitcast(Base::convert_float_to_fp8(flt));
    }

    /// FP16 -> E5M2 conversion - rounds to nearest even
    // static float_e4m3_t from_half(half const& flt);

    // E4M3 -> half
    // static half to_half(float_e4m3_t const& x);

    // E4M3 -> Float
    static float to_float(float_e4m3_t const& x) {
        return Base::convert_fp8_to_float(x.storage);
    }

    //
    // Methods
    //

    /// Default constructor
    float_e4m3_t() = default;

    /// Floating point conversion
    explicit float_e4m3_t(float x) { storage = from_float(x).storage; }

    // explicit float_e4m3_t(half x);

    /// Floating point conversion
    explicit float_e4m3_t(double x) : float_e4m3_t(float(x)) {}

    /// Integer conversion
    explicit float_e4m3_t(int x) : float_e4m3_t(float(x)) {}

    explicit float_e4m3_t(unsigned x) : float_e4m3_t(float(x)) {}

    /// E5M2 conversion. Defined after float_e5m2_t is defined.
    explicit float_e4m3_t(float_e5m2_t x);

    operator float() const { return to_float(*this); }

    /// Converts to half
    // operator half() const;

    /// Converts to float
    explicit operator double() const { return double(to_float(*this)); }

    /// Converts to int
    explicit operator int() const { return int(to_float(*this)); }

    /// Casts to bool

    explicit operator bool() const { return (to_float(*this) != 0.0f); }

    /// Accesses raw internal state
    uint8_t& raw() { return storage; }

    /// Accesses raw internal state
    uint8_t raw() const { return storage; }

    /// Returns the sign bit

    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent
    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) &
                   Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent
    int exponent() const { return exponent_biased() - 15; }

    /// Returns the mantissa

    int mantissa() const { return int(storage & Base::FP8_MANTISSA_MASK); }
};

using fp8_e4m3 = float_e4m3_t;

///////////////////////////////////////////////////////////////
///
/// floating-point 8 type : E5M2
///
///////////////////////////////////////////////////////////////
struct alignas(1) float_e5m2_t : float8_base<FloatEncoding::E5M2> {
    using Base = float8_base<FloatEncoding::E5M2>;
    using Base::Base;

    static float_e5m2_t bitcast(uint8_t x) {
        float_e5m2_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    static float_e5m2_t from_float(float const& flt) {
        return bitcast(Base::convert_float_to_fp8(flt));
    }

    /// FP16 -> E5M2 conversion - rounds to nearest even
    // static float_e5m2_t from_half(half const& flt);

    // E5M2 -> half
    // static half to_half(float_e5m2_t const& x);

    // E5M2 -> Float
    static float to_float(float_e5m2_t const& x) {
        return Base::convert_fp8_to_float(x.storage);
    }

    //
    // Methods
    //

    /// Default constructor
    float_e5m2_t() = default;

    /// Floating point conversion
    explicit float_e5m2_t(float x) { storage = from_float(x).storage; }

    // explicit float_e5m2_t(half x);

    /// Floating point conversion
    explicit float_e5m2_t(double x) : float_e5m2_t(float(x)) {}

    /// Integer conversion
    explicit float_e5m2_t(int x) : float_e5m2_t(float(x)) {}

    explicit float_e5m2_t(unsigned x) : float_e5m2_t(float(x)) {}

    /// E4M3 conversion
    explicit float_e5m2_t(float_e4m3_t x);

    /// Converts to float
    operator float() const { return to_float(*this); }

    /// Converts to half
    // operator half() const;

    /// Converts to float
    explicit operator double() const { return double(to_float(*this)); }

    /// Converts to int
    explicit operator int() const { return int(to_float(*this)); }

    /// Casts to bool
    explicit operator bool() const { return bool(int(to_float(*this))); }

    /// Accesses raw internal state
    uint8_t& raw() { return storage; }

    /// Accesses raw internal state
    uint8_t raw() const { return storage; }

    /// Returns the sign bit
    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent
    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) &
                   Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent
    int exponent() const { return exponent_biased() - 15; }

    /// Returns the mantissa
    int mantissa() const { return int(storage & Base::FP8_MANTISSA_MASK); }
};

using fp8_e5m2 = float_e5m2_t;

}  // namespace ark

// Standard Library operations and definitions for numeric limits
namespace std {
template <>
struct numeric_limits<ark::float_e4m3_t> {
    static bool const is_specialized = true;
    static bool const is_signed = true;
    static bool const is_integer = false;
    static bool const is_exact = false;
    static bool const has_infinity = true;
    static bool const has_quiet_NaN = true;
    static bool const has_signaling_NaN = false;
    static std::float_denorm_style const has_denorm = denorm_present;
    static bool const has_denorm_loss = true;
    static std::float_round_style const round_style = round_to_nearest;
    static bool const is_iec559 = false;
    static bool const is_bounded = true;
    static bool const is_modulo = false;
    static int const digits = 3;

    static ark::float_e4m3_t min() { return ark::float_e4m3_t::bitcast(0x01); }
    static ark::float_e4m3_t lowest() {
        return ark::float_e4m3_t::bitcast(0x7e);
    }
    static ark::float_e4m3_t max() { return ark::float_e4m3_t::bitcast(0x7e); }
    static ark::float_e4m3_t epsilon() {
        return ark::float_e4m3_t::bitcast(0x20);
    }
    static ark::float_e4m3_t round_error() { return ark::float_e4m3_t(0.5f); }
    static ark::float_e4m3_t infinity() {
        return ark::float_e4m3_t::bitcast(ark::float_e4m3_t::FP8_INFINITY_MASK);
    }
    static ark::float_e4m3_t quiet_NaN() {
        return ark::float_e4m3_t::bitcast(ark::float_e4m3_t::FP8_NAN);
    }
    static ark::float_e4m3_t signaling_NaN() {
        return ark::float_e4m3_t::bitcast(ark::float_e4m3_t::FP8_NAN);
    }
    static ark::float_e4m3_t denorm_min() {
        return ark::float_e4m3_t::bitcast(0x01);
    }
};

template <>
struct numeric_limits<ark::float_e5m2_t> {
    static bool const is_specialized = true;
    static bool const is_signed = true;
    static bool const is_integer = false;
    static bool const is_exact = false;
    static bool const has_infinity = true;
    static bool const has_quiet_NaN = true;
    static bool const has_signaling_NaN = false;
    static std::float_denorm_style const has_denorm = denorm_present;
    static bool const has_denorm_loss = true;
    static std::float_round_style const round_style = round_to_nearest;
    static bool const is_iec559 = false;
    static bool const is_bounded = true;
    static bool const is_modulo = false;
    static int const digits = 2;

    static ark::float_e5m2_t min() { return ark::float_e5m2_t::bitcast(0x01); }

    static ark::float_e5m2_t lowest() {
        return ark::float_e5m2_t::bitcast(0xfb);
    }
    static ark::float_e5m2_t max() { return ark::float_e5m2_t::bitcast(0x7b); }

    static ark::float_e5m2_t epsilon() {
        return ark::float_e5m2_t::bitcast(0x34);
    }
    static ark::float_e5m2_t round_error() { return ark::float_e5m2_t(0.5f); }

    static ark::float_e5m2_t infinity() {
        return ark::float_e5m2_t::bitcast(ark::float_e5m2_t::FP8_INFINITY_MASK);
    }
    static ark::float_e5m2_t quiet_NaN() {
        return ark::float_e5m2_t::bitcast(ark::float_e5m2_t::FP8_NAN);
    }
    static ark::float_e5m2_t signaling_NaN() {
        return ark::float_e5m2_t::bitcast(ark::float_e5m2_t::FP8_NAN);
    }
    static ark::float_e5m2_t denorm_min() {
        return ark::float_e5m2_t::bitcast(0x01);
    }
};

}  // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

// Arithmetic operators for float_e4m3_t
bool operator==(float_e4m3_t const& a, float_e4m3_t const& b);
bool operator!=(float_e4m3_t const& a, float_e4m3_t const& b);
bool operator<(float_e4m3_t const& a, float_e4m3_t const& b);
bool operator<=(float_e4m3_t const& a, float_e4m3_t const& b);
bool operator>(float_e4m3_t const& a, float_e4m3_t const& b);
bool operator>=(float_e4m3_t const& a, float_e4m3_t const& b);
float_e4m3_t operator+(float_e4m3_t const& a, float_e4m3_t const& b);
float_e4m3_t operator-(float_e4m3_t const& a, float_e4m3_t const& b);
float_e4m3_t operator*(float_e4m3_t const& a, float_e4m3_t const& b);
float_e4m3_t operator/(float_e4m3_t const& a, float_e4m3_t const& b);
float_e4m3_t& operator+=(float_e4m3_t& a, float_e4m3_t const& b);
float_e4m3_t& operator-=(float_e4m3_t& a, float_e4m3_t const& b);
float_e4m3_t& operator*=(float_e4m3_t& a, float_e4m3_t const& b);
float_e4m3_t& operator/=(float_e4m3_t& a, float_e4m3_t const& b);
float_e4m3_t& operator++(float_e4m3_t& a);
float_e4m3_t& operator--(float_e4m3_t& a);
float_e4m3_t operator++(float_e4m3_t& a, int);
float_e4m3_t operator--(float_e4m3_t& a, int);

// Arithmetic operators for float_e5m2_t
bool operator==(float_e5m2_t const& a, float_e5m2_t const& b);
bool operator!=(float_e5m2_t const& a, float_e5m2_t const& b);
bool operator<(float_e5m2_t const& a, float_e5m2_t const& b);
bool operator<=(float_e5m2_t const& a, float_e5m2_t const& b);
bool operator>(float_e5m2_t const& a, float_e5m2_t const& b);
bool operator>=(float_e5m2_t const& a, float_e5m2_t const& b);
float_e5m2_t operator+(float_e5m2_t const& a, float_e5m2_t const& b);
float_e5m2_t operator-(float_e5m2_t const& a, float_e5m2_t const& b);
float_e5m2_t operator*(float_e5m2_t const& a, float_e5m2_t const& b);
float_e5m2_t operator/(float_e5m2_t const& a, float_e5m2_t const& b);
float_e5m2_t& operator+=(float_e5m2_t& a, float_e5m2_t const& b);
float_e5m2_t& operator-=(float_e5m2_t& a, float_e5m2_t const& b);
float_e5m2_t& operator*=(float_e5m2_t& a, float_e5m2_t const& b);
float_e5m2_t& operator/=(float_e5m2_t& a, float_e5m2_t const& b);
float_e5m2_t& operator++(float_e5m2_t& a);
float_e5m2_t& operator--(float_e5m2_t& a);
float_e5m2_t operator++(float_e5m2_t& a, int);
float_e5m2_t operator--(float_e5m2_t& a, int);

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

ark::float_e4m3_t operator"" _fe4m3(long double x);

ark::float_e4m3_t operator"" _fe4m3(unsigned long long int x);

ark::float_e5m2_t operator"" _fe5m2(long double x);

ark::float_e5m2_t operator"" _fe5m2(unsigned long long int x);

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // ARK_FLOAT8_H_
