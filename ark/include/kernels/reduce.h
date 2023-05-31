// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_REDUCE_H_
#define ARK_KERNELS_REDUCE_H_

#include "transform.h"

namespace ark {

// Static checkers if InShape can be reduced into OutShape.
template <typename InShape, typename OutShape> struct ReduceShapeCheckerW
{
    static_assert(InShape::N == OutShape::N,
                  "Dimension N of input and output do not match");
    static_assert(InShape::C == OutShape::C,
                  "Dimension C of input and output do not match");
    static_assert(InShape::H == OutShape::H,
                  "Dimension H of input and output do not match");
    static_assert(OutShape::W == 1, "Dimension W of output should be 1");
};
template <typename InShape, typename OutShape> struct ReduceShapeCheckerNCH
{
    static_assert(InShape::W == OutShape::W,
                  "Dimension W of input and output do not match");

    // All dimensions should be the same except for one dimension
    static_assert(((OutShape::N == 1) && (InShape::C == OutShape::C) &&
                   (InShape::H == OutShape::H)) ||
                      ((OutShape::C == 1) && (InShape::N == OutShape::N) &&
                       (InShape::H == OutShape::H)) ||
                      ((OutShape::H == 1) && (InShape::N == OutShape::N) &&
                       (InShape::C == OutShape::C)),
                  "Only one dimension can be reduced");
};

struct ReduceTypeSum
{
    template <typename DataType> static DEVICE DataType identity()
    {
        return 0;
    }
    template <typename DataType>
    static DEVICE DataType reduce(const DataType &a, const DataType &b)
    {
        return a + b;
    }
    template <typename DataType>
    static DEVICE DataType postReduce(const DataType &a, int nelem = 1)
    {
        return a;
    }
};

struct ReduceTypeMax
{
    template <typename DataType> static DEVICE DataType identity()
    {
        // TODO: implement
        static_assert(false, "ReduceTypeMax is not implemented");
        return 0;
    }
    template <typename DataType>
    static DEVICE DataType reduce(const DataType &a, const DataType &b)
    {
        return (a > b) ? a : b;
    }
    template <typename DataType>
    static DEVICE DataType postReduce(const DataType &a, int nelem = 1)
    {
        return a;
    }
};

struct ReduceTypeAvg
{
    template <typename DataType> static DEVICE DataType identity()
    {
        return 0;
    }
    template <typename DataType>
    static DEVICE DataType reduce(const DataType &a, const DataType &b)
    {
        return a + b;
    }
    template <typename DataType>
    static DEVICE DataType postReduce(const DataType &a, int nelem = 1)
    {
        return a / nelem;
    }
};

// Shared memory for reduction.
template <typename DataType, int ThreadsNum> struct ReduceSharedStorage
{
    // static const int WarpsNum = ThreadsNum / Arch::ThreadsPerWarp;
    DataType storage[ThreadsNum];
};

// Reduce one dimension of input into output.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, typename ReduceType, typename DataType,
          int NelemPerThread>
struct Reduce
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutShape::W % NelemPerThread == 0,
                  "UnitOutShape::W must be divisible by NelemPerThread");

    // Conduct reduction on W dimension of the input.
    //
    // tn(int): index of the unit operator along the N dimension.
    // tc(int): index of the unit operator along the C dimension.
    // th(int): index of the unit operator along the H dimension.
    // tw(int): index of the unit operator along the W dimension (should be 0).
    static DEVICE void runW(DataType *out, DataType *in, int tn, int tc, int th,
                            int tw)
    {
        using InOutChk = ReduceShapeCheckerW<InShape, OutShape>;

        constexpr int NonReduceDimLen =
            UnitOutShape::N * UnitOutShape::C * UnitOutShape::H;

        // The reduction dimension of the final stage.
        // Assume this division is always exact.
        static_assert((ThreadsNum * NelemPerThread) % NonReduceDimLength == 0);
        constexpr int FinalDim =
            (ThreadsNum * NelemPerThread) / NonReduceDimLength;

        // Shared memory
        ReduceSharedStorage *smem =
            UnitOp::shared_memory<ReduceSharedStorage>();

        int tid = UnitOp::thread_id();
        int tid_w = (tid * NelemPerThread) % FinalDim;
        int tid_h = ((tid * NelemPerThread) / FinalDim) % UnitOutShape::H;
        int tid_c = ((tid * NelemPerThread) / FinalDim / UnitOutShape::H) %
                    UnitOutShape::C;
        int tid_n = (tid * NelemPerThread) / FinalDim / UnitOutShape::H /
                    UnitOutShape::C;

        int idx_out = (tid_h + th * UnitOutShape::H) * OutDims::W +
                      (tid_c + tc * UnitOutShape::C) * OutDims::W * OutDims::H +
                      (tid_n + tn * UnitOutShape::N) * OutDims::W * OutDims::H *
                          OutDims::C;
        int idx_in_base =
            (tid_h + th * UnitOutShape::H) * InDims::W +
            (tid_c + tc * UnitOutShape::C) * InDims::W * InDims::H +
            (tid_n + tn * UnitOutShape::N) * InDims::W * InDims::H * InDims::C;

        smem[tid] = ReduceType::identity<DataType>();

        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += FinalDim) {
            int idx_in = idx_in_base + idx_in_w;
            DataType reduced = in[idx_in];
#pragma unroll
            for (int i = 1; i < NelemPerThread; i++) {
                reduced = ReduceType::reduce(reduced, in[idx_in + i]);
            }
            smem[tid] = ReduceType::reduce(smem[tid], reduced);
        }
        __syncthreads();

        // TODO: final reduction on shared memory using warp shuffle.
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes>
DEVICE void reduce_w_sum(float *c, float *a, float *b, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 2;
    Reduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
           SmemBytes, ReduceTypeSum, float,
           NelemPerThread>::runW(c, a, b, tz / OutShape::C, tz % OutShape::C,
                                 ty, tx);
}

template <bool IsRelu> struct ReduceActivation
{
};

template <> struct ReduceActivation<true>
{
    static DEVICE __half2 compute(__half2 val)
    {
#if __CUDA_ARCH__ >= 800
        return __hmax2(val, (__half2_raw){0, 0});
#else
        float2 fval = __half22float2(val);
        return __floats2half2_rn(fmaxf(fval.x, 0.0f), fmaxf(fval.y, 0.0f));
#endif
    }
};

template <> struct ReduceActivation<false>
{
    static DEVICE __half2 compute(__half2 val)
    {
        return val;
    }
};

template <int M, int N, int K, bool IsRelu, int TN, int SB, int TDM, int TDN,
          int TDK>
struct TransformReduceBatch
{
    static const int MN = M * N;

    static DEVICE __half2 compute(__half2 *x, int midx, int nidx)
    {
        __half *px = &((__half *)x)[midx + nidx * M];
        __half2 sum = *(__half2 *)px;
#pragma unroll
        for (int k = 1; k < K; ++k) {
            sum = __hadd2(sum, *(__half2 *)&px[k * MN]);
        }
        return ReduceActivation<IsRelu>::compute(sum);
    }
};

template <int M, int N, int K, bool IsRelu, int TN, int SB, int TDM, int TDN,
          int TDK>
DEVICE void reduce_batch(ark::half *y, ark::half *x, int tx, int ty, int tz)
{
    using TransformReduceBatch =
        TransformReduceBatch<M, N, K, IsRelu, TN, SB, TDM, TDN, TDK>;
    Transform<TransformReduceBatch, M, N, -1, TN, SB, TDM, TDN, TDK>::run(
        y, x, tx, ty, tz);
}

} // namespace ark

#endif // ARK_KERNELS_REDUCE_H_
