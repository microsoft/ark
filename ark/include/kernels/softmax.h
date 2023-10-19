// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_SOFTMAX_H_
#define ARK_KERNELS_SOFTMAX_H_

#include "reduce.h"

namespace ark {

// Static checkers if InShape can be reduced into OutShape.
template <typename InShape, typename OutShape>
struct SoftmaxShapeChecker {
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
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename DataType, int NelemPerThread>
struct Softmax {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");

    // TODO(chhwang): support NelemPerThread > 1.
    static_assert(NelemPerThread == 1, "Unimplemented");

    static DEVICE void run(DataType *out, const DataType *in, int uop_idx,
                           int smem_per_warp) {
        using InOutChk = SoftmaxShapeChecker<InShape, OutShape>;

        constexpr int NonReduceDimLength = UnitOutDims::NCH;
        // The reduction dimension of the final stage.
        // Assume this division is always exact.
        static_assert(
            (UnitOp::NumThreads * NelemPerThread) % NonReduceDimLength == 0);
        // If we reshape the input into a 2D matrix (NCH x W), NumThreads
        // threads compute NCH rows, and each row's sum is computed by
        // ThreadsPerRow threads. If ThreadsPerRow is larger than warp size, we
        // need to use shared memory to reduce the result of each warp.
        constexpr int ThreadsPerRow =
            (UnitOp::NumThreads * NelemPerThread) / NonReduceDimLength;

        int tid = UnitOp::thread_id();
        int tid_w = (tid * NelemPerThread) % ThreadsPerRow;
        int tid_h = ((tid * NelemPerThread) / ThreadsPerRow) % UnitOutDims::H;
        int tid_c = ((tid * NelemPerThread) / ThreadsPerRow / UnitOutDims::H) %
                    UnitOutDims::C;
        int tid_n = (tid * NelemPerThread) / ThreadsPerRow / UnitOutDims::CH;

        int un = UnitOp::uop_idx_n(uop_idx);
        int uc = UnitOp::uop_idx_c(uop_idx);
        int uh = UnitOp::uop_idx_h(uop_idx);

        int idx_out_base = (tid_h + uh * UnitOutDims::H) * OutDims::W +
                           (tid_c + uc * UnitOutDims::C) * OutDims::HW +
                           (tid_n + un * UnitOutDims::N) * OutDims::CHW;
        int idx_in_base = (tid_h + uh * UnitOutDims::H) * InDims::W +
                          (tid_c + uc * UnitOutDims::C) * InDims::HW +
                          (tid_n + un * UnitOutDims::N) * InDims::CHW;

        // get the max input.
        DataType max_input;
        ReduceTypeMax::identity<1>(&max_input);
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_w;
            ReduceTypeMax::reduce<1>(&max_input, &max_input, &in[idx_in]);
        }

        // final reduction on shared memory using warp shuffle.
        max_input = warpsReduce<ReduceTypeMax, UnitOp, ThreadsPerRow>(
            max_input, tid, smem_per_warp);
        // get the max input.
        ReduceTypeMax::postReduce<1>(&max_input, &max_input, UnitOutDims::W);

        // get the exp input sum, use float to avoid overflow.
        DataType exp_sum_input;
        DataType cmp;
        ReduceTypeSum::identity<1>(&exp_sum_input);
        ReduceTypeSum::identity<1>(&cmp);
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_w;
            DataType val = type::Sub::compute(
                type::Exp::compute(type::Sub::compute(in[idx_in], max_input)),
                cmp);
            DataType tmp = type::Add::compute(exp_sum_input, val);
            cmp =
                type::Sub::compute(type::Sub::compute(tmp, exp_sum_input), val);
            exp_sum_input = tmp;
        }
        exp_sum_input = warpsReduce<ReduceTypeSum, UnitOp, ThreadsPerRow>(
            exp_sum_input, tid, smem_per_warp);
        ReduceTypeSum::postReduce<1>(&exp_sum_input, &exp_sum_input);

        DataType r_exp_sum_input =
            type::Div::compute(type::Cast::compute<DataType>(1), exp_sum_input);

        // the output is
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_w;
            int idx_out = idx_out_base + idx_w;
            out[idx_out] = type::Div::compute(
                type::Exp::compute(type::Sub::compute(in[idx_in], max_input)),
                exp_sum_input);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename DataType>
DEVICE void softmax(DataType *out, DataType *in, int uop_idx,
                    int smem_per_warp) {
    constexpr int NelemPerThread = 1;
    Softmax<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
            SmemBytes, DataType, NelemPerThread>::run(out, in, uop_idx,
                                                      smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_SOFTMAX_H_
