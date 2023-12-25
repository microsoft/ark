// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_LOAD_STORE_H_
#define ARK_KERNELS_LOAD_STORE_H_

#include <cstdint>

#include "device.h"
#include "static_math.h"

namespace ark {

template <int Bytes>
DEVICE void load(void *dst, const void *src) {
    static_assert(math::is_pow2<Bytes>::value || (Bytes % 16 == 0),
                  "Bytes must be a power of 2 or divisible by 16");
    if constexpr (Bytes == 1) {
        *static_cast<uint8_t *>(dst) = *static_cast<const uint8_t *>(src);
    } else if constexpr (Bytes == 2) {
        *static_cast<uint16_t *>(dst) = *static_cast<const uint16_t *>(src);
    } else if constexpr (Bytes == 4) {
        *static_cast<uint32_t *>(dst) = *static_cast<const uint32_t *>(src);
    } else if constexpr (Bytes == 8) {
        *static_cast<uint64_t *>(dst) = *static_cast<const uint64_t *>(src);
    } else {
        uint64_t *pdst = static_cast<uint64_t *>(dst);
        const uint64_t *psrc = static_cast<const uint64_t *>(src);
#pragma unroll
        for (int i = 0; i < Bytes / 16; ++i) {
#if defined(ARK_TARGET_CUDA_ARCH)
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(pdst[2 * i]), "=l"(pdst[2 * i + 1])
                         : "l"(psrc + 2 * i)
                         : "memory");
#else   // !defined(ARK_TARGET_CUDA_ARCH)
            *(static_cast<ulonglong2 *>(dst) + i) =
                *(static_cast<const ulonglong2 *>(src) + i);
#endif  // !defined(ARK_TARGET_CUDA_ARCH)
        }
    }
}

template <int Bytes>
DEVICE void store(void *dst, const void *src) {
    static_assert(math::is_pow2<Bytes>::value || (Bytes % 16 == 0),
                  "Bytes must be a power of 2 or divisible by 16");
    if constexpr (Bytes == 1) {
        *static_cast<uint8_t *>(dst) = *static_cast<const uint8_t *>(src);
    } else if constexpr (Bytes == 2) {
        *static_cast<uint16_t *>(dst) = *static_cast<const uint16_t *>(src);
    } else if constexpr (Bytes == 4) {
        *static_cast<uint32_t *>(dst) = *static_cast<const uint32_t *>(src);
    } else if constexpr (Bytes == 8) {
        *static_cast<uint64_t *>(dst) = *static_cast<const uint64_t *>(src);
    } else {
        uint64_t *pdst = static_cast<uint64_t *>(dst);
        const uint64_t *psrc = static_cast<const uint64_t *>(src);
#pragma unroll
        for (int i = 0; i < Bytes / 16; ++i) {
#if defined(ARK_TARGET_CUDA_ARCH)
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(pdst + 2 * i), "l"(psrc[2 * i]),
                           "l"(psrc[2 * i + 1])
                         : "memory");
#else   // !defined(ARK_TARGET_CUDA_ARCH)
            *(static_cast<ulonglong2 *>(dst) + i) =
                *(static_cast<const ulonglong2 *>(src) + i);
#endif  // !defined(ARK_TARGET_CUDA_ARCH)
        }
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_LOAD_STORE_H_
