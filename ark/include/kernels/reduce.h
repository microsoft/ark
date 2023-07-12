// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_REDUCE_H_
#define ARK_KERNELS_REDUCE_H_

#include "ewise.h"
#include "half.h"
#include "transform.h"
#include <cfloat>
#include <limits>
#include <type_traits>

namespace ark {

typedef enum
{
    N = 0,
    C = 1,
    H = 2,
    W = 3,
} AxisType;

// Shared memory for reduction.
template <typename DataType> struct ReduceSharedStorage
{
    DataType storage[32];
};

/* Reduce single-precision `val` within a single warp. */
template <typename ReduceType, int LanesNum,
          typename DataType = typename ReduceType::DataType>
DEVICE DataType warpReduce(DataType val)
{
    if (LanesNum >= 32) {
        val = ReduceType::reduce(
            val, (DataType)__shfl_xor_sync(0xffffffff, val, 16, 32));
        val = ReduceType::reduce(
            val, (DataType)__shfl_xor_sync(0xffffffff, val, 8, 16));
        val = ReduceType::reduce(
            val, (DataType)__shfl_xor_sync(0xffffffff, val, 4, 8));
        val = ReduceType::reduce(
            val, (DataType)__shfl_xor_sync(0xffffffff, val, 2, 4));
        val = ReduceType::reduce(
            val, (DataType)__shfl_xor_sync(0xffffffff, val, 1, 2));
        return val;
    } else {
        if (LanesNum > 16)
            val = ReduceType::reduce(
                val, (DataType)__shfl_xor_sync(0xffffffff, val, 16, 32));
        if (LanesNum > 8)
            val = ReduceType::reduce(
                val, (DataType)__shfl_xor_sync(0xffffffff, val, 8, 16));
        if (LanesNum > 4)
            val = ReduceType::reduce(
                val, (DataType)__shfl_xor_sync(0xffffffff, val, 4, 8));
        if (LanesNum > 2)
            val = ReduceType::reduce(
                val, (DataType)__shfl_xor_sync(0xffffffff, val, 2, 4));
        if (LanesNum > 1)
            val = ReduceType::reduce(
                val, (DataType)__shfl_xor_sync(0xffffffff, val, 1, 2));
        return val;
    }
}

// Reduce single-precision `val` within multiple warps.
template <typename ReduceType, typename UnitOp, int LanesNum,
          typename DataType = typename ReduceType::DataType>
DEVICE DataType warpsReduce(DataType val, int tid)
{
    val = warpReduce<ReduceType, LanesNum>(val);
    if (LanesNum > 32) {
        ReduceSharedStorage<DataType> *shared =
            UnitOp::template shared_memory<ReduceSharedStorage<DataType>>();
        int laneId = tid & 31;
        int warpId = tid >> 5;
        if (laneId == 0) {
            shared->storage[warpId] = val;
        }
        ark::sync_warps<UnitOp::ThreadsNum>();
        val = (laneId < (LanesNum >> 5))
                  ? shared->storage[laneId]
                  : ReduceType::template identity<DataType>();
        val = warpReduce<ReduceType, 32>(val);
    }

    return val;
}

// Check if InShape can be reduced into OutShape and if UnitOutShape is valid.
template <typename InShape, typename OutShape, typename UnitOutShape, int Axis>
struct ReduceShapeChecker
{
    static_assert((InShape::N == OutShape::N) ||
                      (Axis == AxisType::N && OutShape::N == 1),
                  "Invalid dimension N");
    static_assert((InShape::C == OutShape::C) ||
                      (Axis == AxisType::C && OutShape::C == 1),
                  "Invalid dimension C");
    static_assert((InShape::H == OutShape::H) ||
                      (Axis == AxisType::H && OutShape::H == 1),
                  "Invalid dimension H");
    static_assert((InShape::W == OutShape::W) ||
                      (Axis == AxisType::W && OutShape::W == 1),
                  "Invalid dimension W");
    static_assert((UnitOutShape::N == 1) || (Axis != AxisType::N),
                  "Invalid UnitOutShape::N");
    static_assert((UnitOutShape::C == 1) || (Axis != AxisType::C),
                  "Invalid UnitOutShape::C");
    static_assert((UnitOutShape::H == 1) || (Axis != AxisType::H),
                  "Invalid UnitOutShape::H");
    static_assert((UnitOutShape::W == 1) || (Axis != AxisType::W),
                  "Invalid UnitOutShape::W");
};

template <typename DataType, int NelemPerThread> struct ReduceTypeSum
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void identity(DataType *v)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            v[elem] = 0;
        }
    }
    static DEVICE void reduce(DataType *out, const DataType *in0,
                              const DataType *in1)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            out[elem] = in0[elem] + in1[elem];
        }
    }
    static DEVICE void postReduce(DataType *out, const DataType *in,
                                  int nelem = 1)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            out[elem] = in[elem];
        }
    }
    static DEVICE void singleIdentity(DataType *v)
    {
        *v = 0;
    }
    static DEVICE void singleReduce(DataType *out, const DataType *in0,
                                    const DataType *in1)
    {
        *out = *in0 + *in1;
    }
    static DEVICE void singlePostReduce(DataType *out, const DataType *in,
                                        int nelem = 1)
    {
        *out = *in;
    }
};

template <> struct ReduceTypeSum<half, 2>
{
    using DataType = half;
    using NelemPerThread = 2;

    static DEVICE void identity(half *v)
    {
        *reinterpret_cast<__half2 *>(v) = (__half2_raw){0, 0};
    }
    static DEVICE void reduce(half *out, const half *in0, const half *in1)
    {
        __half2 *out2 = reinterpret_cast<__half2 *>(out);
        __half2 *in02 = reinterpret_cast<__half2 *>(in0);
        __half2 *in12 = reinterpret_cast<__half2 *>(in1);
        *out2 = __hadd2(*in02, *in12);
    }
    static DEVICE void postReduce(half *out, const half *in, int nelem = 1)
    {
        __half2 *out2 = reinterpret_cast<__half2 *>(out);
        __half2 *in2 = reinterpret_cast<__half2 *>(in);
        *out2 = *in2;
    }
};

template <typename DataType, int NelemPerThread> struct ReduceTypeMax
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void identity(DataType *v)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            v[elem] = std::numeric_limits<DataType>::lowest();
        }
    }
    static DEVICE void reduce(DataType *out, const DataType *in0,
                              const DataType *in1)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            out[elem] = (in0[elem] > in1[elem]) ? in0[elem] : in1[elem];
        }
    }
    static DEVICE void postReduce(DataType *out, const DataType *in,
                                  int nelem = 1)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            out[elem] = in[elem];
        }
    }
    static DEVICE void singleIdentity(DataType *v)
    {
        *v = std::numeric_limits<DataType>::lowest();
    }
    static DEVICE void singleReduce(DataType *out, const DataType *in0,
                                    const DataType *in1)
    {
        *out = (*in0 > *in1) ? *in0 : *in1;
    }
    static DEVICE void singlePostReduce(DataType *out, const DataType *in,
                                        int nelem = 1)
    {
        *out = *in;
    }
};

template <> struct ReduceTypeMax<half, 2>
{
    using DataType = half;
    using NelemPerThread = 2;

    static DEVICE void identity(half *v)
    {
        *reinterpret_cast<__half2 *>(v) = (__half2_raw){0xfbff, 0xfbff};
    }
    static DEVICE void reduce(half *out, const half *in0, const half *in1)
    {
#if (__CUDA_ARCH__ >= 800)
        *out = __hmax2(*in0, *in1);
#else
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            out[elem] = (in0[elem] > in1[elem]) ? in0[elem] : in1[elem];
        }
#endif // (__CUDA_ARCH__ >= 800)
    }
    static DEVICE void postReduce(half *out, const half *in, int nelem = 1)
    {
        __half2 *out2 = reinterpret_cast<__half2 *>(out);
        __half2 *in2 = reinterpret_cast<__half2 *>(in);
        *out2 = *in2;
    }
};

template <typename DataType, int NelemPerThread> struct ReduceTypeMean
{
    using DataType = DataType;
    using NelemPerThread = NelemPerThread;

    static DEVICE void identity(DataType *v)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            v[elem] = 0;
        }
    }
    static DEVICE void reduce(DataType *out, const DataType *in0,
                              const DataType *in1)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            out[elem] = in0[elem] + in1[elem];
        }
    }
    static DEVICE void postReduce(DataType *out, const DataType *in,
                                  int nelem = 1)
    {
#pragma unroll
        for (int elem = 0; elem < NelemPerThread; ++elem) {
            out[elem] = in[elem] / nelem;
        }
    }
    static DEVICE void singleIdentity(DataType *v)
    {
        *v = 0;
    }
    static DEVICE void singleReduce(DataType *out, const DataType *in0,
                                    const DataType *in1)
    {
        *out = *in0 + *in1;
    }
    static DEVICE void singlePostReduce(DataType *out, const DataType *in,
                                        int nelem = 1)
    {
        *out = *in / nelem;
    }
};

template <> struct ReduceTypeSum<half, 2>
{
    using DataType = half;
    using NelemPerThread = 2;

    static DEVICE void identity(half *v)
    {
        *reinterpret_cast<__half2 *>(v) = (__half2_raw){0, 0};
    }
    static DEVICE void reduce(half *out, const half *in0, const half *in1)
    {
        __half2 *out2 = reinterpret_cast<__half2 *>(out);
        __half2 *in02 = reinterpret_cast<__half2 *>(in0);
        __half2 *in12 = reinterpret_cast<__half2 *>(in1);
        *out2 = __hadd2(*in02, *in12);
    }
    static DEVICE void postReduce(half *out, const half *in, int nelem = 1)
    {
        __half2 *out2 = reinterpret_cast<__half2 *>(out);
        __half2 *in2 = reinterpret_cast<__half2 *>(in);
        *out2 = __h2div(*in2, __float2half2_rn((float)nelem));
    }
};

// Reduce one dimension of input into output.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, typename ReduceType, int Axis>
struct EwiseReduce
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;
    using DataType = typename ReduceType::DataType;
    using NelemPerThread = typename ReduceType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutShape::W % NelemPerThread == 0,
                  "UnitOutShape::W must be divisible by NelemPerThread");

    // Conduct reduction on N dimension of the input.
    struct EwiseReduceN
    {
        static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                                   int idx_c, int idx_h, int idx_w)
        {
            int idx_out = idx_c * OutDims::HW + idx_h * OutDims::W + idx_w;
            int idx_in = idx_c * InDims::HW + idx_h * InDims::W + idx_w;
            DataType reduced[NelemPerThread];

            ReduceType::identity(reduced);
#pragma unroll
            for (int i = 0; i < InShape::N; ++i) {
                ReduceType::reduce(reduced, reduced,
                                   &in[idx_in + i * InDims::CHW]);
            }
            ReduceType::postReduce(&out[idx_out], reduced, InShape::N);
        }
    }

    // Conduct reduction on C dimension of the input.
    struct EwiseReduceC
    {
        static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                                   int idx_c, int idx_h, int idx_w)
        {
            int idx_out = idx_n * OutDims::CHW + idx_h * OutDims::W + idx_w;
            int idx_in = idx_n * InDims::CHW + idx_h * InDims::W + idx_w;
            DataType reduced[NelemPerThread];

            ReduceType::identity(reduced);
#pragma unroll
            for (int i = 0; i < InShape::C; ++i) {
                ReduceType::reduce(reduced, reduced,
                                   &in[idx_in + i * InDims::HW]);
            }
            ReduceType::postReduce(&out[idx_out], reduced, InShape::C);
        }
    }

    // Conduct reduction on H dimension of the input.
    struct EwiseReduceH
    {
        static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                                   int idx_c, int idx_h, int idx_w)
        {
            int idx_out = idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_w;
            int idx_in = idx_n * InDims::CHW + idx_c * InDims::HW + idx_w;
            DataType reduced[NelemPerThread];

            ReduceType::identity(reduced);
#pragma unroll
            for (int i = 0; i < InShape::H; ++i) {
                ReduceType::reduce(reduced, reduced,
                                   &in[idx_in + i * InDims::W]);
            }
            ReduceType::postReduce(&out[idx_out], reduced, InShape::H);
        }
    }

    // Conduct reduction on W dimension of the input.
    struct EwiseReduceW
    {
        static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                                   int idx_c, int idx_h, int idx_w)
        {
            int idx_out =
                idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W;
            int idx_in =
                idx_n * InDims::CHW + idx_c * InDims::HW + idx_h * InDims::W;
            DataType reduced[NelemPerThread];

            ReduceType::identity(reduced);
#pragma unroll
            for (int i = 0; i < InShape::W; ++i) {
                ReduceType::reduce(reduced, reduced, &in[idx_in + i]);
            }

            DataType finalSum;
            ReduceType::singleIdentity(&finalSum);
#pragma unroll
            for (int i = 0; i < NelemPerThread; ++i) {
                ReduceType::singleReduce(&finalSum, &finalSum, &reduced[i]);
            }
            ReduceType::singlePostReduce(&out[idx_out], &finalSum, InShape::W);
        }
    }

    // Conduct reduction of the input.
    //
    // tn(int): index of the unit operator along the N dimension.
    // tc(int): index of the unit operator along the C dimension.
    // th(int): index of the unit operator along the H dimension.
    // tw(int): index of the unit operator along the W dimension.
    static DEVICE void
    run(DataType *out, DataType *in, int tn, int tc, int th, int tw)
    {
        static_assert(Axis == AxisType::N || Axis == AxisType::C ||
                          Axis == AxisType::H || Axis == AxisType::W,
                      "Invalid reduction axis.");

        using ShapeChecker =
            ReduceShapeChecker<InShape, OutShape, UnitOutShape, Axis>;
        using Reduce = typename std::conditional<
            Axis == AxisType::N, EwiseReduceN,
            typename std::conditional<
                Axis == AxisType::C, EwiseReduceC,
                typename std::conditional<Axis == AxisType::H, EwiseReduceH,
                                          EwiseReduceW>::type>::type>::type;

        Ewise1<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes,
               Reduce>::run(out, in, tn, tc, th, tw);
    }
};

// Warp-wise reduction. Only support reduction along the W dimension.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, typename ReduceType, int Axis>
struct WwiseReduce
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;
    using DataType = typename ReduceType::DataType;
    using NelemPerThread = typename ReduceType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutShape::W % NelemPerThread == 0,
                  "UnitOutShape::W must be divisible by NelemPerThread");
    static_assert(Axis == AxisType::W, "Only support reduction along W axis");

    // TODO(chhwang): support NelemPerThread > 1.
    static_assert(NelemPerThread == 1, "Unimplemented");

    // Conduct reduction on W dimension of the input.
    //
    // tn(int): index of the unit operator along the N dimension.
    // tc(int): index of the unit operator along the C dimension.
    // th(int): index of the unit operator along the H dimension.
    // tw(int): index of the unit operator along the W dimension (should be 0).
    static DEVICE void runW(DataType *out, DataType *in, int tn, int tc, int th,
                            int tw)
    {
        using ShapeChecker =
            ReduceShapeChecker<InShape, OutShape, UnitOutShape, Axis>;

        constexpr int NonReduceDimLength =
            UnitOutShape::N * UnitOutShape::C * UnitOutShape::H;
        // The reduction dimension of the final stage.
        // Assume this division is always exact.
        static_assert((ThreadsNum * NelemPerThread) % NonReduceDimLength == 0);
        // If we reshape the input into a 2D matrix (NCH x W), ThreadsNum
        // threads compute NCH rows, and each row's sum is computed by
        // ThreadsPerRow threads. If ThreadsPerRow is larger than warp size, we
        // need to use shared memory to reduce the result of each warp.
        constexpr int ThreadsPerRow =
            (ThreadsNum * NelemPerThread) / NonReduceDimLength;
        int tid = UnitOp::thread_id();
        int tid_w = (tid * NelemPerThread) % ThreadsPerRow;
        int tid_h = ((tid * NelemPerThread) / ThreadsPerRow) % UnitOutShape::H;
        int tid_c = ((tid * NelemPerThread) / ThreadsPerRow / UnitOutShape::H) %
                    UnitOutShape::C;
        int tid_n = (tid * NelemPerThread) / ThreadsPerRow / UnitOutShape::H /
                    UnitOutShape::C;

        int idx_out = (tid_h + th * UnitOutShape::H) * OutDims::W +
                      (tid_c + tc * UnitOutShape::C) * OutDims::W * OutDims::H +
                      (tid_n + tn * UnitOutShape::N) * OutDims::W * OutDims::H *
                          OutDims::C;
        int idx_in_base =
            (tid_h + th * UnitOutShape::H) * InDims::W +
            (tid_c + tc * UnitOutShape::C) * InDims::W * InDims::H +
            (tid_n + tn * UnitOutShape::N) * InDims::W * InDims::H * InDims::C;

        DataType reduced[NelemPerThread];

        ReduceType::identity(reduced);
#pragma unroll
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            ReduceType::reduce(reduced, reduced, &in[idx_in]);
        }

        DataType finalSum;
        ReduceType::singleIdentity(&finalSum);
#pragma unroll
        for (int i = 0; i < NelemPerThread; ++i) {
            ReduceType::singleReduce(&finalSum, &finalSum, &reduced[i]);
        }

        ark::sync_warps<ThreadsNum>();

        // final reduction on shared memory using warp shuffle.
        finalSum =
            warpsReduce<ReduceType, UnitOp, ThreadsPerRow>(finalSum, tid);

        // write the result to output.
        if (tid % ThreadsPerRow == 0) {
            ReduceType::singlePostReduce(&out[idx_out], &finalSum, InShape::W);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_e_sum(half *out, half *in, int tx, int ty, int tz)
{
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeSum<half, 2>, Axis>::run(out, in,
                                                              tz / OutShape::C,
                                                              tz % OutShape::C,
                                                              ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_e_sum(float *out, float *in, int tx, int ty, int tz)
{
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeSum<float, 1>, Axis>::run(out, in,
                                                               tz / OutShape::C,
                                                               tz % OutShape::C,
                                                               ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_e_mean(half *out, half *in, int tx, int ty, int tz)
{
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMean<half, 2>, Axis>::run(out, in,
                                                               tz / OutShape::C,
                                                               tz % OutShape::C,
                                                               ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_e_mean(float *out, float *in, int tx, int ty, int tz)
{
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMean<float, 1>,
                Axis>::run(out, in, tz / OutShape::C, tz % OutShape::C, ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_e_max(half *out, half *in, int tx, int ty, int tz)
{
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMax<half, 2>, Axis>::run(out, in,
                                                              tz / OutShape::C,
                                                              tz % OutShape::C,
                                                              ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_e_max(float *out, float *in, int tx, int ty, int tz)
{
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMax<float, 1>, Axis>::run(out, in,
                                                               tz / OutShape::C,
                                                               tz % OutShape::C,
                                                               ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_w_sum(half *out, half *in, int tx, int ty, int tz)
{
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeSum<half, 1>, Axis>::runW(out, in,
                                                               tz / OutShape::C,
                                                               tz % OutShape::C,
                                                               ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_w_sum(float *out, float *in, int tx, int ty, int tz)
{
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeSum<float, 1>, Axis>::runW(out, in,
                                                                tz /
                                                                    OutShape::C,
                                                                tz %
                                                                    OutShape::C,
                                                                ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_w_mean(half *out, half *in, int tx, int ty, int tz)
{
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMean<half, 1>, Axis>::runW(out, in,
                                                                tz /
                                                                    OutShape::C,
                                                                tz %
                                                                    OutShape::C,
                                                                ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_w_mean(float *out, float *in, int tx, int ty, int tz)
{
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMean<float, 1>,
                Axis>::runW(out, in, tz / OutShape::C, tz % OutShape::C, ty,
                            tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_w_max(half *out, half *in, int tx, int ty, int tz)
{
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMax<half, 1>, Axis>::runW(out, in,
                                                               tz / OutShape::C,
                                                               tz % OutShape::C,
                                                               ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, int Axis>
DEVICE void reduce_w_max(float *out, float *in, int tx, int ty, int tz)
{
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
                SmemBytes, ReduceTypeMax<float, 1>, Axis>::runW(out, in,
                                                                tz /
                                                                    OutShape::C,
                                                                tz %
                                                                    OutShape::C,
                                                                ty, tx);
}

} // namespace ark

#endif // ARK_KERNELS_REDUCE_H_
