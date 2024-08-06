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

#include "float8.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace ark {

///////////////////////////////////////////////////////////////////////////////////////////////////

// Arithmetic operators for float_e4m3_t
bool operator==(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float(a) == float(b);
}

bool operator!=(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float(a) != float(b);
}

bool operator<(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float(a) < float(b);
}

bool operator<=(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float(a) <= float(b);
}

bool operator>(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float(a) > float(b);
}

bool operator>=(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float(a) >= float(b);
}
float_e4m3_t operator+(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float_e4m3_t(float(a) + float(b));
}

float_e4m3_t operator-(float_e4m3_t const& a) {
    return float_e4m3_t(-float(a));
}

float_e4m3_t operator-(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float_e4m3_t(float(a) - float(b));
}

float_e4m3_t operator*(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float_e4m3_t(float(a) * float(b));
}

float_e4m3_t operator/(float_e4m3_t const& a, float_e4m3_t const& b) {
    return float_e4m3_t(float(a) / float(b));
}

float_e4m3_t& operator+=(float_e4m3_t& a, float_e4m3_t const& b) {
    a = float_e4m3_t(float(a) + float(b));
    return a;
}

float_e4m3_t& operator-=(float_e4m3_t& a, float_e4m3_t const& b) {
    a = float_e4m3_t(float(a) - float(b));
    return a;
}

float_e4m3_t& operator*=(float_e4m3_t& a, float_e4m3_t const& b) {
    a = float_e4m3_t(float(a) * float(b));
    return a;
}

float_e4m3_t& operator/=(float_e4m3_t& a, float_e4m3_t const& b) {
    a = float_e4m3_t(float(a) / float(b));
    return a;
}

float_e4m3_t& operator++(float_e4m3_t& a) {
    float tmp(a);
    ++tmp;
    a = float_e4m3_t(tmp);
    return a;
}

float_e4m3_t& operator--(float_e4m3_t& a) {
    float tmp(a);
    --tmp;
    a = float_e4m3_t(tmp);
    return a;
}

float_e4m3_t operator++(float_e4m3_t& a, int) {
    float_e4m3_t ret(a);
    float tmp(a);
    tmp++;
    a = float_e4m3_t(tmp);
    return ret;
}

float_e4m3_t operator--(float_e4m3_t& a, int) {
    float_e4m3_t ret(a);
    float tmp(a);
    tmp--;
    a = float_e4m3_t(tmp);
    return ret;
}

// Arithmetic operators for float_e5m2_t
bool operator==(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float(a) == float(b);
}

bool operator!=(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float(a) != float(b);
}

bool operator<(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float(a) < float(b);
}

bool operator<=(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float(a) <= float(b);
}

bool operator>(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float(a) > float(b);
}

bool operator>=(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float(a) >= float(b);
}

float_e5m2_t operator+(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float_e5m2_t(float(a) + float(b));
}

float_e5m2_t operator-(float_e5m2_t const& a) {
    return float_e5m2_t(-float(a));
}

float_e5m2_t operator-(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float_e5m2_t(float(a) - float(b));
}

float_e5m2_t operator*(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float_e5m2_t(float(a) * float(b));
}

float_e5m2_t operator/(float_e5m2_t const& a, float_e5m2_t const& b) {
    return float_e5m2_t(float(a) / float(b));
}

float_e5m2_t& operator+=(float_e5m2_t& a, float_e5m2_t const& b) {
    a = float_e5m2_t(float(a) + float(b));
    return a;
}

float_e5m2_t& operator-=(float_e5m2_t& a, float_e5m2_t const& b) {
    a = float_e5m2_t(float(a) - float(b));
    return a;
}

float_e5m2_t& operator*=(float_e5m2_t& a, float_e5m2_t const& b) {
    a = float_e5m2_t(float(a) * float(b));
    return a;
}

float_e5m2_t& operator/=(float_e5m2_t& a, float_e5m2_t const& b) {
    a = float_e5m2_t(float(a) / float(b));
    return a;
}

float_e5m2_t& operator++(float_e5m2_t& a) {
    float tmp(a);
    ++tmp;
    a = float_e5m2_t(tmp);
    return a;
}

float_e5m2_t& operator--(float_e5m2_t& a) {
    float tmp(a);
    --tmp;
    a = float_e5m2_t(tmp);
    return a;
}

float_e5m2_t operator++(float_e5m2_t& a, int) {
    float_e5m2_t ret(a);
    float tmp(a);
    tmp++;
    a = float_e5m2_t(tmp);
    return ret;
}

float_e5m2_t operator--(float_e5m2_t& a, int) {
    float_e5m2_t ret(a);
    float tmp(a);
    tmp--;
    a = float_e5m2_t(tmp);
    return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace ark

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

ark::float_e4m3_t operator"" _fe4m3(long double x) {
    return ark::float_e4m3_t(float(x));
}

ark::float_e4m3_t operator"" _fe4m3(unsigned long long int x) {
    return ark::float_e4m3_t(int(x));
}

ark::float_e5m2_t operator"" _fe5m2(long double x) {
    return ark::float_e5m2_t(float(x));
}

ark::float_e5m2_t operator"" _fe5m2(unsigned long long int x) {
    return ark::float_e5m2_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////