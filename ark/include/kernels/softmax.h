// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_SOFTMAX_H_
#define ARK_KERNELS_SOFTMAX_H_

#include "reduce.h"

namespace ark {

// Static checkers if InShape can be reduced into OutShape.
template <typename InShape, typename OutShape> struct SoftmaxShapeChecker
{
    static_assert(InShape::N == OutShape::N,
                  "Dimension N of input and output do not match");
    static_assert(InShape::C == OutShape::C,
                  "Dimension C of input and output do not match");
    static_assert(InShape::H == OutShape::H,
                  "Dimension H of input and output do not match");
    static_assert(OutShape::W == OutShape::W,
                  "Dimension W of input and output do not match");
};

// Perform layer normalization on input and write the result on output.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename DataType, int NelemPerThread>
struct Softmax
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");

    // TODO(chhwang): support NelemPerThread > 1.
    static_assert(NelemPerThread == 1, "Unimplemented");

    static DEVICE void run(DataType *out, const DataType *in, int tn, int tc,
                           int th, int tw)
    {
        using InOutChk = SoftmaxShapeChecker<InShape, OutShape>;
        using ReduceTypeMax = ReduceTypeMax<DataType, NelemPerThread>;
        using ReduceTypeSum = ReduceTypeSum<DataType, NelemPerThread>;

        constexpr int NonReduceDimLength = UnitOutDims::NCH;
        // The reduction dimension of the final stage.
        // Assume this division is always exact.
        static_assert((NumThreads * NelemPerThread) % NonReduceDimLength == 0);
        // If we reshape the input into a 2D matrix (NCH x W), NumThreads
        // threads compute NCH rows, and each row's sum is computed by
        // ThreadsPerRow threads. If ThreadsPerRow is larger than warp size, we
        // need to use shared memory to reduce the result of each warp.
        constexpr int ThreadsPerRow =
            (NumThreads * NelemPerThread) / NonReduceDimLength;

        int tid = UnitOp::thread_id();
        int tid_w = (tid * NelemPerThread) % ThreadsPerRow;
        int tid_h = ((tid * NelemPerThread) / ThreadsPerRow) % UnitOutDims::H;
        int tid_c = ((tid * NelemPerThread) / ThreadsPerRow / UnitOutDims::H) %
                    UnitOutDims::C;
        int tid_n = (tid * NelemPerThread) / ThreadsPerRow / UnitOutDims::CH;

        int idx_in_base = (tid_h + th * UnitOutDims::H) * InDims::W +
                          (tid_c + tc * UnitOutDims::C) * InDims::HW +
                          (tid_n + tn * UnitOutDims::N) * InDims::CHW;

        // get the max input.
        DataType max_input;
        ReduceTypeMax::singleIdentity(&max_input);
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            ReduceTypeMax::singleReduce(&max_input, &max_input, &in[idx_in]);
        }
        ark::sync_warps<NumThreads>();

        // final reduction on shared memory using warp shuffle.
        max_input =
            warpsReduce<ReduceTypeMax, UnitOp, ThreadsPerRow>(max_input, tid);
        // get the max input.
        ReduceTypeMax::singlePostReduce(&max_input, &max_input, UnitOutDims::W);

        // get the exp input sum, use float to avoid overflow.
        DataType exp_sum_input;
        ReduceTypeSum::singleIdentity(&exp_sum_input);
        ark::sync_warps<NumThreads>();
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            exp_sum_input = exp_sum_input + expf(in[idx_in] - max_input);
        }
        ark::sync_warps<NumThreads>();
        exp_sum_input = warpsReduce<ReduceTypeSum, UnitOp, ThreadsPerRow>(
            exp_sum_input, tid);
        ReduceTypeSum::singlePostReduce(&exp_sum_input, &exp_sum_input);
        ark::sync_warps<NumThreads>();
        // the output is
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            out[idx_in] = expf(in[idx_in] - max_input) / exp_sum_input;
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void softmax(float *out, float *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    Softmax<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
            SmemBytes, float, NelemPerThread>::run(out, in, tz / OutShape::C,
                                                   tz % OutShape::C, ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void softmax(ark::half *out, ark::half *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    Softmax<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
            SmemBytes, ark::half, NelemPerThread>::run(out, in,
                                                       tz / OutShape::C,
                                                       tz % OutShape::C, ty,
                                                       tx);
}

} // namespace ark

#endif // ARK_KERNELS_SOFTMAX_H_
