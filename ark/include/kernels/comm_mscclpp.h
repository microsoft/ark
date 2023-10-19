// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMM_MSLL_H_
#define ARK_KERNELS_COMM_MSLL_H_

#include <cuda_runtime.h>

#include <msll/proxy_channel_device.hpp>
#include <msll/sm_channel_device.hpp>

#include "common.h"
#include "ewise.h"
#include "unit_op.h"

extern __constant__ msll::SimpleProxyChannelDeviceHandle _ARK_PROXY_CHANS[];
extern __constant__ msll::SmChannelDeviceHandle _ARK_SM_CHANS[];

namespace ark {
namespace comm {

template <typename OutDims, typename DataType>
struct MsllEwiseSumCompType {};

template <int NBytes>
union BytesPack {};

template <>
union BytesPack<16> {
    uint16_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
};

template <>
union BytesPack<8> {
    uint16_t u16[4];
    uint32_t u32[2];
};

DEVICE void load(BytesPack<16> &v, const longlong2 *p) {
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(v.u64[0]), "=l"(v.u64[1])
                 : "l"(p)
                 : "memory");
}

DEVICE void load(BytesPack<8> &v, const uint2 *p) {
    asm volatile("ld.volatile.global.v2.b32 {%0,%1}, [%2];"
                 : "=r"(v.u32[0]), "=r"(v.u32[1])
                 : "l"(p)
                 : "memory");
}

DEVICE void store(longlong2 *p, const BytesPack<16> &v) {
    asm volatile("st.volatile.global.v2.b64 [%0], {%1,%2};"
                 :
                 : "l"(p), "l"(v.u64[0]), "l"(v.u64[1])
                 : "memory");
}

DEVICE void store(uint2 *p, const BytesPack<8> &v) {
    asm volatile("st.volatile.global.v2.b32 [%0], {%1,%2};"
                 :
                 : "l"(p), "r"(v.u32[0]), "r"(v.u32[1])
                 : "memory");
}

DEVICE void add_half8(BytesPack<16> &dst, BytesPack<16> &src) {
    __half2 *pd = reinterpret_cast<__half2 *>(dst.u32);
    __half2 *ps = reinterpret_cast<__half2 *>(src.u32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        union {
            __half2 h2; uint32_t u32;
        } d, s;
        d.h2 = pd[i];
        s.h2 = ps[i];
        pd[i] = __hadd2(d.h2, s.h2);
    }
}

DEVICE void add_half4(BytesPack<8> &dst, BytesPack<8> &src) {
    __half *pd = reinterpret_cast<__half *>(dst.u16);
    __half *ps = reinterpret_cast<__half *>(src.u16);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        __half2 d, s;
        d.x = pd[i * 2];
        d.y = pd[i * 2 + 1];
        s.x = ps[i * 2];
        s.y = ps[i * 2 + 1];
        d = __hadd2(d, s);
        pd[i * 2] = d.x;
        pd[i * 2 + 1] = d.y;
    }
}
}  // namespace comm
}  // namespace ark

#endif  // ARK_KERNELS_COMM_H_
