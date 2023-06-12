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
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, typename DataType, int NelemPerThread>
struct Softmax
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static DEVICE void run(DataType *out, DataType *in, int tn, int tc, int th,
                           int tw)
    {
        using InOutChk = SoftmaxShapeChecker<InShape, OutShape>;

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

        int idx_in_base =
            (tid_h + th * UnitOutShape::H) * InDims::W +
            (tid_c + tc * UnitOutShape::C) * InDims::W * InDims::H +
            (tid_n + tn * UnitOutShape::N) * InDims::W * InDims::H * InDims::C;

        // get the max input.
        DataType max_input = ReduceTypeMax::template identity<DataType>();
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            max_input = ReduceTypeMax::reduce(max_input, in[idx_in]);
        }
        ark::sync_warps<ThreadsNum>();

        // final reduction on shared memory using warp shuffle.
        max_input = warpsReduce<ReduceTypeMax, DataType, UnitOp, ThreadsPerRow,
                                ThreadsNum>(max_input, tid);
        // get the max input.
        max_input = ReduceTypeMax::postReduce(max_input, UnitOutShape::W);

        // get the exp input sum, use float to avoid overflow.
        DataType exp_sum_input = ReduceTypeSum::template identity<DataType>();
        ark::sync_warps<ThreadsNum>();
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            exp_sum_input = exp_sum_input + expf(in[idx_in] - max_input);
        }
        ark::sync_warps<ThreadsNum>();
        exp_sum_input =
            warpsReduce<ReduceTypeSum, DataType, UnitOp, ThreadsPerRow,
                        ThreadsNum>(exp_sum_input, tid);
        exp_sum_input = ReduceTypeSum::postReduce(exp_sum_input);
        ark::sync_warps<ThreadsNum>();
        // the output is
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            out[idx_in] = expf(in[idx_in] - max_input) / exp_sum_input;
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes>
DEVICE void softmax(float *out, float *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    Softmax<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
            SmemBytes, float, NelemPerThread>::run(out, in, tz / OutShape::C,
                                                   tz % OutShape::C, ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes>
DEVICE void softmax(ark::half *out, ark::half *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    Softmax<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
            SmemBytes, ark::half, NelemPerThread>::run(out, in,
                                                       tz / OutShape::C,
                                                       tz % OutShape::C, ty,
                                                       tx);
}

} // namespace ark

#endif // ARK_KERNELS_SOFTMAX_H_
