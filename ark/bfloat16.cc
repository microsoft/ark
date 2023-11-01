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

/// Borrowing CUTLASS's host-side half_t until we can move on to C++23 and use
/// std::bfloat16_t

#include "bfloat16.h"

namespace ark {

/// Assignment from half_t
template <>
bfloat16_t& bfloat16_t::operator=(bfloat16_t const& x) {
    storage = x.storage;
    return *this;
}

/// Assignment from float
template <>
bfloat16_t& bfloat16_t::operator=(float const& x) {
    storage = bfloat16_t(x).storage;
    return *this;
}

bool signbit(ark::bfloat16_t const& h) { return h.signbit(); }

ark::bfloat16_t abs(ark::bfloat16_t const& h) {
    return ark::bfloat16_t::bitcast(h.raw() & 0x7fffffff);
}

bool isnan(ark::bfloat16_t const& h) {
    return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

bool isfinite(ark::bfloat16_t const& h) {
    return (h.exponent_biased() != 0x0ff);
}

ark::bfloat16_t nan_bf16(const char*) {
    // NVIDIA canonical NaN
    return ark::bfloat16_t::bitcast(0x7fff);
}

bool isinf(ark::bfloat16_t const& h) {
    return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

bool isnormal(ark::bfloat16_t const& h) {
    return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

int fpclassify(ark::bfloat16_t const& h) {
    int exp = h.exponent_biased();
    int mantissa = h.mantissa();
    if (exp == 0x0ff) {
        if (mantissa) {
            return FP_NAN;
        } else {
            return FP_INFINITE;
        }
    } else if (!exp) {
        if (mantissa) {
            return FP_SUBNORMAL;
        } else {
            return FP_ZERO;
        }
    }
    return FP_NORMAL;
}

ark::bfloat16_t sqrt(ark::bfloat16_t const& h) {
    return ark::bfloat16_t(std::sqrt(float(h)));
}

bfloat16_t copysign(bfloat16_t const& a, bfloat16_t const& b) {
    uint16_t a_bits;
    uint16_t b_bits;

    std::memcpy(&a_bits, &a, sizeof(a_bits));
    std::memcpy(&b_bits, &b, sizeof(b_bits));

    uint16_t a_mag = (a_bits & 0x7fff);
    uint16_t b_sign = (b_bits & 0x8000);
    uint16_t result = (a_mag | b_sign);

    return bfloat16_t::bitcast(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

namespace std {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

ark::bfloat16_t abs(ark::bfloat16_t const& h) { return ark::abs(h); }

ark::bfloat16_t max(ark::bfloat16_t const& a, ark::bfloat16_t const& b) {
    return ark::bfloat16_t(std::max(float(a), float(b)));
}

ark::bfloat16_t exp(ark::bfloat16_t const& h) {
    return ark::bfloat16_t(std::exp(float(h)));
}

}  // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

bool operator==(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return float(lhs) == float(rhs);
}

bool operator!=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return float(lhs) != float(rhs);
}

bool operator<(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return float(lhs) < float(rhs);
}

bool operator<=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return float(lhs) <= float(rhs);
}

bool operator>(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return float(lhs) > float(rhs);
}

bool operator>=(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return float(lhs) >= float(rhs);
}

bfloat16_t operator+(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return bfloat16_t(float(lhs) + float(rhs));
}

bfloat16_t operator-(bfloat16_t const& lhs) { return bfloat16_t(-float(lhs)); }

bfloat16_t operator-(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return bfloat16_t(float(lhs) - float(rhs));
}

bfloat16_t operator*(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return bfloat16_t(float(lhs) * float(rhs));
}

bfloat16_t operator/(bfloat16_t const& lhs, bfloat16_t const& rhs) {
    return bfloat16_t(float(lhs) / float(rhs));
}

bfloat16_t& operator+=(bfloat16_t& lhs, bfloat16_t const& rhs) {
    lhs = bfloat16_t(float(lhs) + float(rhs));
    return lhs;
}

bfloat16_t& operator-=(bfloat16_t& lhs, bfloat16_t const& rhs) {
    lhs = bfloat16_t(float(lhs) - float(rhs));
    return lhs;
}

bfloat16_t& operator*=(bfloat16_t& lhs, bfloat16_t const& rhs) {
    lhs = bfloat16_t(float(lhs) * float(rhs));
    return lhs;
}

bfloat16_t& operator/=(bfloat16_t& lhs, bfloat16_t const& rhs) {
    lhs = bfloat16_t(float(lhs) / float(rhs));
    return lhs;
}

bfloat16_t& operator++(bfloat16_t& lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = bfloat16_t(tmp);
    return lhs;
}

bfloat16_t& operator--(bfloat16_t& lhs) {
    float tmp(lhs);
    --tmp;
    lhs = bfloat16_t(tmp);
    return lhs;
}

bfloat16_t operator++(bfloat16_t& lhs, int) {
    bfloat16_t ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = bfloat16_t(tmp);
    return ret;
}

bfloat16_t operator--(bfloat16_t& lhs, int) {
    bfloat16_t ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = bfloat16_t(tmp);
    return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

ark::bfloat16_t operator"" _bf16(long double x) {
    return ark::bfloat16_t(float(x));
}

ark::bfloat16_t operator"" _bf16(unsigned long long int x) {
    return ark::bfloat16_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
