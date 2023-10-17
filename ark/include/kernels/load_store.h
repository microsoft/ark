// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_LOAD_STORE_H_
#define ARK_KERNELS_LOAD_STORE_H_

#include <cstdint>
#include "device.h"

namespace ark {

DEVICE longlong2 load_128b(const longlong2 *p) {
    longlong2 v;
#if defined(ARK_TARGET_CUDA_ARCH)
    asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                    : "=l"(v.x), "=l"(v.y)
                    : "l"(p)
                    : "memory");
#else
    v.x = p->x;
    v.y = p->y;
#endif
    return v;
};

DEVICE void store_128b(longlong2 *p, const longlong2 &v) {
#if defined(ARK_TARGET_CUDA_ARCH)
    asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                :
                : "l"(p), "l"(v.x), "l"(v.y)
                : "memory");
#else
    p->x = v.x;
    p->y = v.y;
#endif
}

}  // namespace ark

#endif  // ARK_KERNELS_LOAD_STORE_H_
