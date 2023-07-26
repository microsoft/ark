// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EWISE_H_
#define ARK_KERNELS_EWISE_H_

#include "static_math.h"
#include "unit_op.h"

namespace ark {

// Element-wise computation operator with a single input.
template <typename OutDims, typename OutShape, typename UnitOutDims,
          int NumThreads, int SmemBytes, typename CompType>
struct Ewise1
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;
    using DataType = typename CompType::DataType;
    static const int NelemPerThread = CompType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

    // Conduct element-wise computation on input and write the result on output.
    //
    // tn(int): index of the unit operator along the N dimension.
    // tc(int): index of the unit operator along the C dimension.
    // th(int): index of the unit operator along the H dimension.
    // tw(int): index of the unit operator along the W dimension.
    static DEVICE void run(DataType *out, DataType *in, int tn, int tc, int th,
                           int tw)
    {
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

            CompType::compute(out, in, tid_n + tn * UnitOutDims::N,
                              tid_c + tc * UnitOutDims::C,
                              tid_h + th * UnitOutDims::H,
                              tid_w + tw * UnitOutDims::W);
        }
    }
};

} // namespace ark

#endif // ARK_KERNELS_EWISE_H_
