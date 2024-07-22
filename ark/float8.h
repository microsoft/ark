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
    static bool isfinite(float flt);

    /// Is NaN implementation
    static bool isnan(float flt);

    /// Is infinite implementation
    static bool isinf(float flt);

    /// FP32 -> FP8 conversion - rounds to nearest even
    static uint8_t convert_float_to_fp8(float const& flt);

    /// Converts a fp8 value stored as a uint8_t to a float
    static float convert_fp8_to_float(uint8_t const& x);
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

    static float_e4m3_t bitcast(uint8_t x);

    /// FP32 -> FP8 conversion - rounds to nearest even

    static float_e4m3_t from_float(float const& flt);

    /// FP16 -> E5M2 conversion - rounds to nearest even

    //static float_e4m3_t from_half(half const& flt);

    // E4M3 -> half

    //static half to_half(float_e4m3_t const& x);

    // E4M3 -> Float

    static float to_float(float_e4m3_t const& x);

    //
    // Methods
    //

    /// Constructor inheritance

    /// Default constructor
    float_e4m3_t() = default;

    /// Floating point conversion

    explicit float_e4m3_t(float x);

    //explicit float_e4m3_t(half x);
    /// Floating point conversion

    explicit float_e4m3_t(double x);

    /// Integer conversion

    explicit float_e4m3_t(int x);

    explicit float_e4m3_t(unsigned x);

    /// E5M2 conversion. Defined after float_e5m2_t is defined.

    explicit float_e4m3_t(float_e5m2_t x);

    operator float() const;

    /// Converts to half

    //operator half() const;

    /// Converts to float

    explicit operator double() const;
    /// Converts to int

    explicit operator int() const;

    /// Casts to bool

    explicit operator bool() const;

    /// Accesses raw internal state

    uint8_t& raw();
    /// Accesses raw internal state

    uint8_t raw() const;

    /// Returns the sign bit

    bool signbit() const;

    /// Returns the biased exponent

    int exponent_biased() const;
    /// Returns the unbiased exponent

    int exponent() const;

    /// Returns the mantissa

    int mantissa() const;
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

    static float_e5m2_t bitcast(uint8_t x);

    /// FP32 -> FP8 conversion - rounds to nearest even
    static float_e5m2_t from_float(float const& flt);

    /// FP16 -> E5M2 conversion - rounds to nearest even
    //static float_e5m2_t from_half(half const& flt);

    // E5M2 -> half
    //static half to_half(float_e5m2_t const& x);

    // E5M2 -> Float
    static float to_float(float_e5m2_t const& x);

    //
    // Methods
    //

    /// Default constructor
    float_e5m2_t() = default;

    /// Floating point conversion
    explicit float_e5m2_t(float x);

    explicit float_e5m2_t(half x);

    /// Floating point conversion
    explicit float_e5m2_t(double x);

    /// Integer conversion
    explicit float_e5m2_t(int x);

    explicit float_e5m2_t(unsigned x);

    /// E4M3 conversion
    explicit float_e5m2_t(float_e4m3_t x);

    /// Converts to float
    operator float() const;

    /// Converts to half
    //operator half() const;

    /// Converts to float
    explicit operator double() const;

    /// Converts to int
    explicit operator int() const;

    /// Casts to bool
    explicit operator bool() const;

    /// Accesses raw internal state
    uint8_t& raw();

    /// Accesses raw internal state
    uint8_t raw() const;

    /// Returns the sign bit
    bool signbit() const;

    /// Returns the biased exponent
    int exponent_biased() const;

    /// Returns the unbiased exponent
    int exponent() const;

    /// Returns the mantissa
    int mantissa() const;
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


ark::float_e4m3_t operator "" _fe4m3(long double x);


ark::float_e4m3_t operator "" _fe4m3(unsigned long long int x);


ark::float_e5m2_t operator "" _fe5m2(long double x);


ark::float_e5m2_t operator "" _fe5m2(unsigned long long int x);

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // ARK_FLOAT8_H_
