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
/// std::float16_t

#include "half.h"

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Assignment from half_t
template <>
half_t& half_t::operator=(half_t const& x) {
    storage = x.storage;
    return *this;
}

/// Assignment from float
template <>
half_t& half_t::operator=(float const& x) {
    storage = half_t::convert(x).storage;
    return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

bool signbit(ark::half_t const& h) { return ((h.raw() & 0x8000) != 0); }

ark::half_t abs(ark::half_t const& h) {
    return ark::half_t::bitcast(h.raw() & 0x7fff);
}

bool isnan(ark::half_t const& h) {
    return (h.exponent_biased() == 0x1f) && h.mantissa();
}

bool isfinite(ark::half_t const& h) { return (h.exponent_biased() != 0x1f); }

ark::half_t nanh(const char*) {
    // NVIDIA canonical NaN
    return ark::half_t::bitcast(0x7fff);
}

bool isinf(ark::half_t const& h) {
    return (h.exponent_biased() == 0x1f) && !h.mantissa();
}

bool isnormal(ark::half_t const& h) {
    return h.exponent_biased() && h.exponent_biased() != 0x1f;
}

int fpclassify(ark::half_t const& h) {
    int exp = h.exponent_biased();
    int mantissa = h.mantissa();
    if (exp == 0x1f) {
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

ark::half_t sqrt(ark::half_t const& h) {
    return ark::half_t(std::sqrt(float(h)));
}

half_t copysign(half_t const& a, half_t const& b) {
    uint16_t a_mag = (a.raw() & 0x7fff);
    uint16_t b_sign = (b.raw() & 0x8000);
    uint16_t result = (a_mag | b_sign);

    return half_t::bitcast(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace std {

ark::half_t abs(ark::half_t const& h) { return ark::abs(h); }

ark::half_t max(ark::half_t const& a, ark::half_t const& b) {
    return ark::half_t(std::max(float(a), float(b)));
}

ark::half_t exp(ark::half_t const& h) {
    return ark::half_t(std::exp(float(h)));
}

}  // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

bool operator==(half_t const& lhs, half_t const& rhs) {
    return float(lhs) == float(rhs);
}

bool operator!=(half_t const& lhs, half_t const& rhs) {
    return float(lhs) != float(rhs);
}

bool operator<(half_t const& lhs, half_t const& rhs) {
    return float(lhs) < float(rhs);
}

bool operator<=(half_t const& lhs, half_t const& rhs) {
    return float(lhs) <= float(rhs);
}

bool operator>(half_t const& lhs, half_t const& rhs) {
    return float(lhs) > float(rhs);
}

bool operator>=(half_t const& lhs, half_t const& rhs) {
    return float(lhs) >= float(rhs);
}

half_t operator+(half_t const& lhs, half_t const& rhs) {
    return half_t(float(lhs) + float(rhs));
}

half_t operator-(half_t const& lhs) { return half_t(-float(lhs)); }

half_t operator-(half_t const& lhs, half_t const& rhs) {
    return half_t(float(lhs) - float(rhs));
}

half_t operator*(half_t const& lhs, half_t const& rhs) {
    return half_t(float(lhs) * float(rhs));
}

half_t operator/(half_t const& lhs, half_t const& rhs) {
    return half_t(float(lhs) / float(rhs));
}

half_t& operator+=(half_t& lhs, half_t const& rhs) {
    lhs = half_t(float(lhs) + float(rhs));
    return lhs;
}

half_t& operator-=(half_t& lhs, half_t const& rhs) {
    lhs = half_t(float(lhs) - float(rhs));
    return lhs;
}

half_t& operator*=(half_t& lhs, half_t const& rhs) {
    lhs = half_t(float(lhs) * float(rhs));
    return lhs;
}

half_t& operator/=(half_t& lhs, half_t const& rhs) {
    lhs = half_t(float(lhs) / float(rhs));
    return lhs;
}

half_t& operator++(half_t& lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = half_t(tmp);
    return lhs;
}

half_t& operator--(half_t& lhs) {
    float tmp(lhs);
    --tmp;
    lhs = half_t(tmp);
    return lhs;
}

half_t operator++(half_t& lhs, int) {
    half_t ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = half_t(tmp);
    return ret;
}

half_t operator--(half_t& lhs, int) {
    half_t ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = half_t(tmp);
    return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

ark::half_t operator"" _hf(long double x) { return ark::half_t(float(x)); }

ark::half_t operator"" _hf(unsigned long long int x) {
    return ark::half_t(int(x));
}
