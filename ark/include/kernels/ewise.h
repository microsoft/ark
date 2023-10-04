// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EWISE_H_
#define ARK_KERNELS_EWISE_H_

#include "common.h"

namespace ark {

/// Element-wise computation operator with a single input.
template <typename OutDims, typename OutShape, typename UnitOutDims,
          int NumThreads, int SmemBytes, typename CompType>
struct Ewise1 {
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;
    using DataType = typename CompType::DataType;
    static const int NelemPerThread = CompType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

    /// Conduct element-wise computation on input and write the result on
    /// output.
    /// @param out Output data.
    /// @param in Input data.
    /// @param uop_idx Index of the unit operator.
    static DEVICE void run(DataType *out, DataType *in, int uop_idx) {
        int un = UnitOp::uop_idx_n(uop_idx);
        int uc = UnitOp::uop_idx_c(uop_idx);
        int uh = UnitOp::uop_idx_h(uop_idx);
        int uw = UnitOp::uop_idx_w(uop_idx);

        for (int tid = UnitOp::thread_id();; tid += NumThreads) {
            int tid_w = (tid * NelemPerThread) % UnitOutDims::W;
            int tid_h =
                ((tid * NelemPerThread) / UnitOutDims::W) % UnitOutDims::H;
            int tid_c =
                ((tid * NelemPerThread) / UnitOutDims::HW) % UnitOutDims::C;
            int tid_n = (tid * NelemPerThread) / UnitOutDims::CHW;

            if (tid_n >= UnitOutDims::N) {
                break;
            }

            int idx_n = tid_n + un * UnitOutDims::N;
            int idx_c = tid_c + uc * UnitOutDims::C;
            int idx_h = tid_h + uh * UnitOutDims::H;
            int idx_w = tid_w + uw * UnitOutDims::W;

            if (idx_n >= OutShape::N || idx_c >= OutShape::C ||
                idx_h >= OutShape::H || idx_w >= OutShape::W) {
                continue;
            }

            CompType::compute(out, in, idx_n, idx_c, idx_h, idx_w);
        }
    }
};

}  // namespace ark

#endif  // ARK_KERNELS_EWISE_H_
