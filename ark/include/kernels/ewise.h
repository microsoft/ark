// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EWISE_H_
#define ARK_KERNELS_EWISE_H_

#include "static_math.h"
#include "unit_op.h"

namespace ark {

// Element-wise computation operator with a single input.
template <typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes,
          typename CompType>
struct Ewise1
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;
    using DataType = typename CompType::DataType;

    static_assert(CompType::NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutShape::W % CompType::NelemPerThread == 0,
                  "UnitOutShape::W must be divisible by NelemPerThread");

    // Conduct element-wise computation on input and write the result on output.
    //
    // tn(int): index of the unit operator along the N dimension.
    // tc(int): index of the unit operator along the C dimension.
    // th(int): index of the unit operator along the H dimension.
    // tw(int): index of the unit operator along the W dimension.
    static DEVICE void run(DataType *out, DataType *in, int tn, int tc, int th,
                           int tw)
    {
        for (int tid = UnitOp::thread_id();; tid += ThreadsNum) {
            int tid_w = (tid * CompType::NelemPerThread) % UnitOutShape::W;
            int tid_h =
                ((tid * CompType::NelemPerThread) / UnitOutShape::W) % UnitOutShape::H;
            int tid_c =
                ((tid * CompType::NelemPerThread) / UnitOutShape::HW) %
                UnitOutShape::C;
            int tid_n = (tid * CompType::NelemPerThread) / UnitOutShape::CHW;

            if (tid_n >= UnitOutShape::N) {
                break;
            }

            CompType::compute(
                out, in, tid_n + tn * UnitOutShape::N,
                tid_c + tc * UnitOutShape::C, tid_h + th * UnitOutShape::H,
                tid_w + tw * UnitOutShape::W);
        }
    }
};

} // namespace ark

#endif // ARK_KERNELS_EWISE_H_
