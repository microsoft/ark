// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ARITHMETIC_H_
#define ARK_KERNELS_ARITHMETIC_H_

#include "broadcast.h"
#include "transform.h"

namespace ark {

template <typename In0Shape, typename In1Shape> struct Add
{
    template <int NelemPerThread, typename DataType>
    static DEVICE void compute(DataType *c, DataType *a, DataType *b)
    {
        *c = *a + *b;
        if (In0Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = *a + b[i];
            }
        } else if (In1Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = a[i] + *b;
            }
        } else {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = a[i] + b[i];
            }
        }
    }

    template <int NelemPerThread>
    static DEVICE void compute(ark::half *c, ark::half *a, ark::half *b)
    {
        static_assert(NelemPerThread % 2 == 0,
                      "NelemPerThread must be a multiple of 2");
        constexpr int Nhalf2 = NelemPerThread / 2;
        __half2 *pc = (__half2 *)c;
        __half2 *pa = (__half2 *)a;
        __half2 *pb = (__half2 *)b;
        *pc = __hadd2(*pa, *pb);
        if (In0Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < Nhalf2; ++i) {
                pc[i] = __hadd2(*pa, pb[i]);
            }
        } else if (In1Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < Nhalf2; ++i) {
                pc[i] = __hadd2(pa[i], *pb);
            }
        } else {
#pragma unroll
            for (int i = 1; i < Nhalf2; ++i) {
                pc[i] = __hadd2(pa[i], pb[i]);
            }
        }
    }
};

template <typename In0Shape, typename In1Shape> struct Mul
{
    template <int NelemPerThread, typename DataType>
    static DEVICE void compute(DataType *c, DataType *a, DataType *b)
    {
        *c = *a * *b;
        if (In0Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = *a * b[i];
            }
        } else if (In1Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = a[i] * *b;
            }
        } else {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = a[i] * b[i];
            }
        }
    }

    template <int NelemPerThread>
    static DEVICE void compute(ark::half *c, ark::half *a, ark::half *b)
    {
        static_assert(NelemPerThread % 2 == 0,
                      "NelemPerThread must be a multiple of 2");
        constexpr int Nhalf2 = NelemPerThread / 2;
        __half2 *pc = (__half2 *)c;
        __half2 *pa = (__half2 *)a;
        __half2 *pb = (__half2 *)b;
        *pc = __hmul2(*pa, *pb);
        if (In0Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < Nhalf2; ++i) {
                pc[i] = __hmul2(*pa, pb[i]);
            }
        } else if (In1Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < Nhalf2; ++i) {
                pc[i] = __hmul2(pa[i], *pb);
            }
        } else {
#pragma unroll
            for (int i = 1; i < Nhalf2; ++i) {
                pc[i] = __hmul2(pa[i], pb[i]);
            }
        }
    }
};

template <int M> struct TransformScale
{
    static DEVICE __half2 compute(__half2 *a, __half2 *b, int midx, int nidx)
    {
        return __hmul2(*(__half2 *)&((__half *)a)[midx + nidx * M], *b);
    }
};

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes>
DEVICE void add(float *c, float *a, float *b, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    Broadcast<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
              UnitOutShape, ThreadsNum, SmemBytes, Add<In0Shape, In1Shape>,
              float, NelemPerThread>::run(c, a, b, tz / OutShape::C,
                                          tz % OutShape::C, ty, tx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes>
DEVICE void add(ark::half *c, ark::half *a, ark::half *b, int tx, int ty,
                int tz)
{
    constexpr int NelemPerThread = 2;
    Broadcast<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
              UnitOutShape, ThreadsNum, SmemBytes, Add<In0Shape, In1Shape>,
              ark::half, NelemPerThread>::run(c, a, b, tz / OutShape::C,
                                              tz % OutShape::C, ty, tx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes>
DEVICE void mul(float *c, float *a, float *b, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    Broadcast<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
              UnitOutShape, ThreadsNum, SmemBytes, Mul<In0Shape, In1Shape>,
              float, NelemPerThread>::run(c, a, b, tz / OutShape::C,
                                          tz % OutShape::C, ty, tx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes>
DEVICE void mul(ark::half *c, ark::half *a, ark::half *b, int tx, int ty,
                int tz)
{
    constexpr int NelemPerThread = 2;
    Broadcast<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
              UnitOutShape, ThreadsNum, SmemBytes, Mul<In0Shape, In1Shape>,
              ark::half, NelemPerThread>::run(c, a, b, tz / OutShape::C,
                                              tz % OutShape::C, ty, tx);
}

template <int M, int N, int TN, int SB, int TDM, int TDN, int TDK, int VAL>
DEVICE void scale(ark::half *y, ark::half *x, float val, int tx, int ty, int tz)
{
    __half2 val2 = __float2half2_rn(val);
    Transform<TransformScale<M>, M, N, 1, TN, SB, TDM, TDN, TDK>::run(
        y, x, (ark::half *)&val2, tx, ty, tz);
}

} // namespace ark

#endif // ARK_KERNELS_ARITHMETIC_H_
