// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_LOAD_STORE_H_
#define ARK_KERNELS_LOAD_STORE_H_

#include <cstdint>

#include "atomic.h"
#include "device.h"
#include "static_math.h"

namespace ark {

template <typename T, bool Atomic = false>
DEVICE void load_t(void *dst, const void *src) {
    static_assert(math::is_pow2<sizeof(T)>::value,
                  "type size must be a power of 2");
    static_assert(sizeof(T) <= 16, "type size must be <= 16");
    if constexpr (Atomic) {
        if constexpr (sizeof(T) == 16) {
#if defined(ARK_TARGET_CUDA_ARCH)
            uint64_t *pdst = reinterpret_cast<uint64_t *>(dst);
            const uint64_t *psrc = reinterpret_cast<const uint64_t *>(src);
            asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(pdst[0]), "=l"(pdst[1])
                         : "l"(psrc)
                         : "memory");
#else   // !defined(ARK_TARGET_CUDA_ARCH)
            ulonglong2 *pdst = reinterpret_cast<ulonglong2 *>(dst);
            const ulonglong2 *psrc = reinterpret_cast<const ulonglong2 *>(src);
            pdst->x = atomicLoadRelaxed(&psrc->x);
            pdst->y = atomicLoadRelaxed(&psrc->y);
#endif  // !defined(ARK_TARGET_CUDA_ARCH)
        } else {
            *reinterpret_cast<T *>(dst) =
                atomicLoadRelaxed(reinterpret_cast<const T *>(src));
        }
    } else {
        *reinterpret_cast<T *>(dst) = *reinterpret_cast<const T *>(src);
    }
}

template <int Bytes, bool Atomic = false>
DEVICE void load(void *dst, const void *src) {
    static_assert(math::is_pow2<Bytes>::value || (Bytes % 16 == 0),
                  "Bytes must be a power of 2 or divisible by 16");
    if constexpr (Bytes == 1) {
        load_t<uint8_t, Atomic>(dst, src);
    } else if constexpr (Bytes == 2) {
        load_t<uint16_t, Atomic>(dst, src);
    } else if constexpr (Bytes == 4) {
        load_t<uint32_t, Atomic>(dst, src);
    } else if constexpr (Bytes == 8) {
        load_t<uint64_t, Atomic>(dst, src);
    } else {
        ulonglong2 *pdst = static_cast<ulonglong2 *>(dst);
        const ulonglong2 *psrc = static_cast<const ulonglong2 *>(src);
#pragma unroll
        for (int i = 0; i < Bytes / 16; ++i) {
            load_t<ulonglong2, Atomic>(pdst + i, psrc + i);
        }
    }
}

template <typename T, bool Atomic = false>
DEVICE void store_t(void *dst, const void *src) {
    static_assert(math::is_pow2<sizeof(T)>::value,
                  "type size must be a power of 2");
    static_assert(sizeof(T) <= 16, "type size must be <= 16");
    if constexpr (Atomic) {
        if constexpr (sizeof(T) == 16) {
#if defined(ARK_TARGET_CUDA_ARCH)
            const uint64_t *psrc = reinterpret_cast<const uint64_t *>(src);
            uint64_t *pdst = reinterpret_cast<uint64_t *>(dst);
            asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(pdst), "l"(psrc[0]), "l"(psrc[1])
                         : "memory");
#else   // !defined(ARK_TARGET_CUDA_ARCH)
            const ulonglong2 *psrc = reinterpret_cast<const ulonglong2 *>(src);
            ulonglong2 *pdst = reinterpret_cast<ulonglong2 *>(dst);
            atomicStoreRelaxed(&pdst->x, psrc->x);
            atomicStoreRelaxed(&pdst->y, psrc->y);
#endif  // !defined(ARK_TARGET_CUDA_ARCH)
        } else {
            atomicStoreRelaxed(reinterpret_cast<T *>(dst),
                               *reinterpret_cast<const T *>(src));
        }
    } else {
        *reinterpret_cast<T *>(dst) = *reinterpret_cast<const T *>(src);
    }
}

template <int Bytes, bool Atomic = false>
DEVICE void store(void *dst, const void *src) {
    static_assert(math::is_pow2<Bytes>::value || (Bytes % 16 == 0),
                  "Bytes must be a power of 2 or divisible by 16");
    if constexpr (Bytes == 1) {
        store_t<uint8_t, Atomic>(dst, src);
    } else if constexpr (Bytes == 2) {
        store_t<uint16_t, Atomic>(dst, src);
    } else if constexpr (Bytes == 4) {
        store_t<uint32_t, Atomic>(dst, src);
    } else if constexpr (Bytes == 8) {
        store_t<uint64_t, Atomic>(dst, src);
    } else {
        const ulonglong2 *psrc = static_cast<const ulonglong2 *>(src);
        ulonglong2 *pdst = static_cast<ulonglong2 *>(dst);
#pragma unroll
        for (int i = 0; i < Bytes / 16; ++i) {
            store_t<ulonglong2, Atomic>(pdst + i, psrc + i);
        }
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_LOAD_STORE_H_
