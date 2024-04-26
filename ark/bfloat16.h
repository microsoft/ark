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

#ifndef ARK_BFLOAT16_H_
#define ARK_BFLOAT16_H_

/// Borrowing CUTLASS's host-side half_t until we can move on to C++23 and use
/// std::bfloat16_t

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point type with 8 bits of exponent and 7 bits of mantissa.
struct alignas(2) bfloat16_t {
    //
    // Data members
    //

    /// Storage type
    uint16_t storage;

    //
    // Methods
    //

    /// Constructs from an unsigned short
    static bfloat16_t bitcast(uint16_t x) {
        bfloat16_t h;
        h.storage = x;
        return h;
    }

    /// Default constructor
    bfloat16_t() = default;

    /// Floating-point conversion - round toward nearest
    explicit bfloat16_t(float x) {
        uint32_t bits;

        std::memcpy(&bits, &x, sizeof(bits));

        if ((bits & 0x7f800000) != 0x7f800000) {
            bool mantissa_bit = ((bits & (1 << 16)) != 0);
            bool round_bit = ((bits & (1 << 15)) != 0);
            bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);

            if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
                bits += uint32_t(1 << 16);
            }
        } else if (bits & ~0xff800000) {
            bits = 0x7fffffff;
        }

        storage = uint16_t((bits >> 16) & 0xffff);
    }

    /// Floating-point conversion - round toward nearest
    explicit bfloat16_t(double x) : bfloat16_t(float(x)) {}

    /// Integer conversion - round toward nearest
    explicit bfloat16_t(int x) {
        float flt = static_cast<float>(x);
        uint32_t bits;

        std::memcpy(&bits, &flt, sizeof(bits));

        storage = uint16_t(bits >> 16);
    }

    /// Converts to float
    operator float() const {
        unsigned bits = (unsigned(storage) << 16);
        float flt;
        std::memcpy(&flt, &bits, sizeof(flt));
        return flt;
    }

    /// Converts to float
    explicit operator double() const { return double(float(*this)); }

    /// Converts to int
    explicit operator int() const { return int(float(*this)); }

    /// Casts to bool
    explicit operator bool() const { return (float(*this) != 0.0f); }

    /// Assignment
    template <typename T>
    bfloat16_t& operator=(T const& x) {
        storage = bfloat16_t(float(x)).storage;
        return *this;
    }

    /// Obtains raw bits
    uint16_t raw() const { return storage; }
    /// Returns the sign bit
    bool signbit() const { return ((raw() & 0x8000) != 0); }

    /// Returns the biased exponent
    int exponent_biased() const { return int((raw() >> 7) & 0x0ff); }

    /// Returns the unbiased exponent
    int exponent() const { return exponent_biased() - 127; }

    /// Returns the mantissa
    int mantissa() const { return int(raw() & 0x7f); }
};

using bf16 = bfloat16_t;

/// Assignment from half_t
template <>
bfloat16_t& bfloat16_t::operator=(bfloat16_t const& x);

/// Assignment from float
template <>
bfloat16_t& bfloat16_t::operator=(float const& x);

///////////////////////////////////////////////////////////////////////////////////////////////////

bool signbit(ark::bfloat16_t const& h);

ark::bfloat16_t abs(ark::bfloat16_t const& h);

bool isnan(ark::bfloat16_t const& h);

bool isfinite(ark::bfloat16_t const& h);

ark::bfloat16_t nan_bf16(const char*);

bool isinf(ark::bfloat16_t const& h);

bool isnormal(ark::bfloat16_t const& h);

int fpclassify(ark::bfloat16_t const& h);

ark::bfloat16_t sqrt(ark::bfloat16_t const& h);

bfloat16_t copysign(bfloat16_t const& a, bfloat16_t const& b);

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
struct numeric_limits<ark::bfloat16_t> {
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
    static bool const is_iec559 = false;
    static bool const is_bounded = true;
    static bool const is_modulo = false;
    static int const digits = 7;

    /// Least positive value
    static ark::bfloat16_t min() { return ark::bfloat16_t::bitcast(0x01); }

    /// Minimum finite value
    static ark::bfloat16_t lowest() { return ark::bfloat16_t::bitcast(0xff7f); }

    /// Maximum finite value
    static ark::bfloat16_t max() { return ark::bfloat16_t::bitcast(0x7f7f); }

    /// Returns smallest finite value
    static ark::bfloat16_t epsilon() {
        return ark::bfloat16_t::bitcast(0x1000);
    }

    /// Returns smallest finite value
    static ark::bfloat16_t round_error() { return ark::bfloat16_t(0.5f); }

    /// Returns smallest finite value
    static ark::bfloat16_t infinity() {
        return ark::bfloat16_t::bitcast(0x7f80);
    }

    /// Returns smallest finite value
    static ark::bfloat16_t quiet_NaN() {
        return ark::bfloat16_t::bitcast(0x7fff);
    }

    /// Returns smallest finite value
    static ark::bfloat16_t signaling_NaN() {
        return ark::bfloat16_t::bitcast(0x7fff);
    }

    /// Returns smallest finite value
    static ark::bfloat16_t denorm_min() {
        return ark::bfloat16_t::bitcast(0x1);
    }
};

ark::bfloat16_t abs(ark::bfloat16_t const& h);

ark::bfloat16_t max(ark::bfloat16_t const& a, ark::bfloat16_t const& b);

ark::bfloat16_t exp(ark::bfloat16_t const& h);

}  // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

bool operator==(bfloat16_t const& lhs, bfloat16_t const& rhs);

bool operator!=(bfloat16_t const& lhs, bfloat16_t const& rhs);

bool operator<(bfloat16_t const& lhs, bfloat16_t const& rhs);

bool operator<=(bfloat16_t const& lhs, bfloat16_t const& rhs);

bool operator>(bfloat16_t const& lhs, bfloat16_t const& rhs);

bool operator>=(bfloat16_t const& lhs, bfloat16_t const& rhs);

bfloat16_t operator+(bfloat16_t const& lhs, bfloat16_t const& rhs);

bfloat16_t operator-(bfloat16_t const& lhs);

bfloat16_t operator-(bfloat16_t const& lhs, bfloat16_t const& rhs);

bfloat16_t operator*(bfloat16_t const& lhs, bfloat16_t const& rhs);

bfloat16_t operator/(bfloat16_t const& lhs, bfloat16_t const& rhs);

bfloat16_t& operator+=(bfloat16_t& lhs, bfloat16_t const& rhs);

bfloat16_t& operator-=(bfloat16_t& lhs, bfloat16_t const& rhs);

bfloat16_t& operator*=(bfloat16_t& lhs, bfloat16_t const& rhs);

bfloat16_t& operator/=(bfloat16_t& lhs, bfloat16_t const& rhs);

bfloat16_t& operator++(bfloat16_t& lhs);

bfloat16_t& operator--(bfloat16_t& lhs);

bfloat16_t operator++(bfloat16_t& lhs, int);

bfloat16_t operator--(bfloat16_t& lhs, int);

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

ark::bfloat16_t operator"" _bf16(long double x);

ark::bfloat16_t operator"" _bf16(unsigned long long int x);

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // ARK_BFLOAT16_H_
