// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ARITHMETIC_H_
#define ARK_KERNELS_ARITHMETIC_H_

#include "common.h"

namespace ark {

struct Add
{
    static DEVICE float compute(float a, float b)
    {
        return a + b;
    }
    static DEVICE half compute(half a, half b)
    {
        return a + b;
    }
    static DEVICE __half compute(__half a, __half b)
    {
        return __hadd(a, b);
    }
    static DEVICE __half2 compute(__half2 a, __half2 b)
    {
        return __hadd2(a, b);
    }
};

struct Sub
{
    static DEVICE float compute(float a, float b)
    {
        return a - b;
    }
    static DEVICE half compute(half a, half b)
    {
        return a - b;
    }
    static DEVICE __half compute(__half a, __half b)
    {
        return __hsub(a, b);
    }
    static DEVICE __half2 compute(__half2 a, __half2 b)
    {
        return __hsub2(a, b);
    }
};

struct Mul
{
    static DEVICE float compute(float a, float b)
    {
        return a * b;
    }
    static DEVICE half compute(half a, half b)
    {
        return a * b;
    }
    static DEVICE __half compute(__half a, __half b)
    {
        return __hmul(a, b);
    }
    static DEVICE __half2 compute(__half2 a, __half2 b)
    {
        return __hmul2(a, b);
    }
};

struct Div
{
    static DEVICE float compute(float a, float b)
    {
        return a / b;
    }
    static DEVICE half compute(half a, half b)
    {
        return a / b;
    }
    static DEVICE __half compute(__half a, __half b)
    {
        return __hdiv(a, b);
    }
    static DEVICE __half2 compute(__half2 a, __half2 b)
    {
        return __h2div(a, b);
    }
};

template <typename _ArithmeticType, typename _In0Shape, typename _In1Shape,
          typename _DataType, int _NelemPerThread>
struct Arithmetic
{
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *c, const DataType *a,
                               const DataType *b)
    {
        *c = *a + *b;
        if (_In0Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = _ArithmeticType::compute(*a, b[i]);
            }
        } else if (_In1Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = _ArithmeticType::compute(a[i], *b);
            }
        } else {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = _ArithmeticType::compute(a[i], b[i]);
            }
        }
    }
};

template <typename _ArithmeticType, typename _In0Shape, typename _In1Shape>
struct Arithmetic<_ArithmeticType, _In0Shape, _In1Shape, float, 2>
{
    using DataType = float;
    static const int NelemPerThread = 2;

    static DEVICE void compute(float *c, const float *a, const float *b)
    {
        float2 *pc = (float2 *)c;
        if (_In0Shape::W == 1) {
            float2 *pb = (float2 *)b;
            pc->x = _ArithmeticType::compute(*a, pb->x);
            pc->y = _ArithmeticType::compute(*a, pb->y);
        } else if (_In1Shape::W == 1) {
            float2 *pa = (float2 *)a;
            pc->x = _ArithmeticType::compute(pa->x, *b);
            pc->y = _ArithmeticType::compute(pa->y, *b);
        } else {
            float2 *pa = (float2 *)a;
            float2 *pb = (float2 *)b;
            pc->x = _ArithmeticType::compute(pa->x, pb->x);
            pc->y = _ArithmeticType::compute(pa->y, pb->y);
        }
    }
};

template <typename _ArithmeticType, typename _In0Shape, typename _In1Shape>
struct Arithmetic<_ArithmeticType, _In0Shape, _In1Shape, float, 4>
{
    using DataType = float;
    static const int NelemPerThread = 4;

    static DEVICE void compute(float *c, const float *a, const float *b)
    {
        if (_In0Shape::W == 1) {
            longlong2 reg_b;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            float4 *pb = (float4 *)&reg_b;
            float4 *pc = (float4 *)&reg_c;
            float v = *a;
            pc->w = _ArithmeticType::compute(v, pb->w);
            pc->x = _ArithmeticType::compute(v, pb->x);
            pc->y = _ArithmeticType::compute(v, pb->y);
            pc->z = _ArithmeticType::compute(v, pb->z);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else if (_In1Shape::W == 1) {
            longlong2 reg_a;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            float4 *pa = (float4 *)&reg_a;
            float4 *pc = (float4 *)&reg_c;
            float v = *b;
            pc->w = _ArithmeticType::compute(pa->w, v);
            pc->x = _ArithmeticType::compute(pa->x, v);
            pc->y = _ArithmeticType::compute(pa->y, v);
            pc->z = _ArithmeticType::compute(pa->z, v);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else {
            longlong2 reg_a;
            longlong2 reg_b;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            float4 *pa = (float4 *)&reg_a;
            float4 *pb = (float4 *)&reg_b;
            float4 *pc = (float4 *)&reg_c;
            pc->w = _ArithmeticType::compute(pa->w, pb->w);
            pc->x = _ArithmeticType::compute(pa->x, pb->x);
            pc->y = _ArithmeticType::compute(pa->y, pb->y);
            pc->z = _ArithmeticType::compute(pa->z, pb->z);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        }
    }
};

template <typename _ArithmeticType, typename _In0Shape, typename _In1Shape>
struct Arithmetic<_ArithmeticType, _In0Shape, _In1Shape, half, 2>
{
    using DataType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *c, const half *a, const half *b)
    {
        __half2 *pc = (__half2 *)c;
        if (_In0Shape::W == 1) {
            __half2 *pb = (__half2 *)b;
            *pc =
                _ArithmeticType::compute(__half2half2(*(const __half *)a), *pb);
        } else if (_In1Shape::W == 1) {
            __half2 *pa = (__half2 *)a;
            *pc =
                _ArithmeticType::compute(*pa, __half2half2(*(const __half *)b));
        } else {
            __half2 *pa = (__half2 *)a;
            __half2 *pb = (__half2 *)b;
            *pc = _ArithmeticType::compute(*pa, *pb);
        }
    }
};

template <typename _ArithmeticType, typename _In0Shape, typename _In1Shape>
struct Arithmetic<_ArithmeticType, _In0Shape, _In1Shape, half, 4>
{
    using DataType = half;
    static const int NelemPerThread = 4;

    static DEVICE void compute(half *c, const half *a, const half *b)
    {
        if (_In0Shape::W == 1) {
            uint64_t reg_b = *(uint64_t *)b;
            uint64_t reg_c;
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)a);
            pc[0] = _ArithmeticType::compute(v, pb[0]);
            pc[1] = _ArithmeticType::compute(v, pb[1]);
            *(uint64_t *)c = reg_c;
        } else if (_In1Shape::W == 1) {
            uint64_t reg_a = *(uint64_t *)a;
            uint64_t reg_c;
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)b);
            pc[0] = _ArithmeticType::compute(pa[0], v);
            pc[1] = _ArithmeticType::compute(pa[1], v);
            *(uint64_t *)c = reg_c;
        } else {
            uint64_t reg_a = *(uint64_t *)a;
            uint64_t reg_b = *(uint64_t *)b;
            uint64_t reg_c;
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            pc[0] = _ArithmeticType::compute(pa[0], pb[0]);
            pc[1] = _ArithmeticType::compute(pa[1], pb[1]);
            *(uint64_t *)c = reg_c;
        }
    }
};

template <typename _ArithmeticType, typename _In0Shape, typename _In1Shape>
struct Arithmetic<_ArithmeticType, _In0Shape, _In1Shape, half, 8>
{
    using DataType = half;
    static const int NelemPerThread = 8;

    static DEVICE void compute(half *c, const half *a, const half *b)
    {
        if (_In0Shape::W == 1) {
            longlong2 reg_b;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)a);
            pc[0] = _ArithmeticType::compute(v, pb[0]);
            pc[1] = _ArithmeticType::compute(v, pb[1]);
            pc[2] = _ArithmeticType::compute(v, pb[2]);
            pc[3] = _ArithmeticType::compute(v, pb[3]);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else if (_In1Shape::W == 1) {
            longlong2 reg_a;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)b);
            pc[0] = _ArithmeticType::compute(pa[0], v);
            pc[1] = _ArithmeticType::compute(pa[1], v);
            pc[2] = _ArithmeticType::compute(pa[2], v);
            pc[3] = _ArithmeticType::compute(pa[3], v);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else {
            longlong2 reg_a;
            longlong2 reg_b;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            pc[0] = _ArithmeticType::compute(pa[0], pb[0]);
            pc[1] = _ArithmeticType::compute(pa[1], pb[1]);
            pc[2] = _ArithmeticType::compute(pa[2], pb[2]);
            pc[3] = _ArithmeticType::compute(pa[3], pb[3]);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        }
    }
};

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void add(float *c, const float *a, const float *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 4 == 0)   ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Add, In0Shape, In1Shape, float,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void add(half *c, const half *a, const half *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 8 == 0)   ? 8
                                   : (UnitOutDims::W % 4 == 0) ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Add, In0Shape, In1Shape, half,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void sub(float *c, const float *a, const float *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 4 == 0)   ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Sub, In0Shape, In1Shape, float,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void sub(half *c, const half *a, const half *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 8 == 0)   ? 8
                                   : (UnitOutDims::W % 4 == 0) ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Sub, In0Shape, In1Shape, half,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void mul(float *c, float *a, float *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 4 == 0)   ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Mul, In0Shape, In1Shape, float,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void mul(half *c, half *a, half *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 8 == 0)   ? 8
                                   : (UnitOutDims::W % 4 == 0) ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Mul, In0Shape, In1Shape, half,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void div(float *c, float *a, float *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 4 == 0)   ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Div, In0Shape, In1Shape, float,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void div(half *c, half *a, half *b, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 8 == 0)   ? 8
                                   : (UnitOutDims::W % 4 == 0) ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Div, In0Shape, In1Shape, half,
                          NelemPerThread>>::run(c, a, b, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void scale(half *y, half *x, float val, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 8 == 0)   ? 8
                                   : (UnitOutDims::W % 4 == 0) ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    half val_h(val);
    using ValDims = Vec<1, 1, 1, 1>;
    using ValShape = Vec<1, 1, 1, 1>;
    Broadcast2<
        InDims, InShape, ValDims, ValShape, OutDims, OutShape, UnitOutDims,
        NumThreads, SmemBytes,
        Arithmetic<Mul, InShape, ValShape, half, NelemPerThread>>::run(y, x,
                                                                       &val_h,
                                                                       uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void scale(float *y, float *x, float val, int uop_idx, int)
{
    constexpr int NelemPerThread = (UnitOutDims::W % 4 == 0)   ? 4
                                   : (UnitOutDims::W % 2 == 0) ? 2
                                                               : 1;
    using ValDims = Vec<1, 1, 1, 1>;
    using ValShape = Vec<1, 1, 1, 1>;
    Broadcast2<InDims, InShape, ValDims, ValShape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes,
               Arithmetic<Mul, InShape, ValShape, float,
                          NelemPerThread>>::run(y, x, &val, uop_idx);
}

} // namespace ark

#endif // ARK_KERNELS_ARITHMETIC_H_
