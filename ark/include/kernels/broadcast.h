// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BROADCAST_H_
#define ARK_KERNELS_BROADCAST_H_

#include "static_math.h"
#include "unit_op.h"
#include "vec.h"

namespace ark {

// Static checker if In0Shape and In1Shape can be broadcasted into OutShape.
template <typename In0Shape, typename In1Shape, typename OutShape>
struct BroadcastShapeChecker
{
    static_assert(In0Shape::N == 1 || In1Shape::N == 1 ||
                      In0Shape::N == In1Shape::N,
                  "Cannot broadcast dimension N of inputs");
    static_assert(In0Shape::C == 1 || In1Shape::C == 1 ||
                      In0Shape::C == In1Shape::C,
                  "Cannot broadcast dimension C of inputs");
    static_assert(In0Shape::H == 1 || In1Shape::H == 1 ||
                      In0Shape::H == In1Shape::H,
                  "Cannot broadcast dimension H of inputs");
    static_assert(In0Shape::W == 1 || In1Shape::W == 1 ||
                      In0Shape::W == In1Shape::W,
                  "Cannot broadcast dimension W of inputs");

    // Derived OutShape.
    using DerOutShape = Vec<math::max<In0Shape::N, In1Shape::N>::value,
                            math::max<In0Shape::C, In1Shape::C>::value,
                            math::max<In0Shape::H, In1Shape::H>::value,
                            math::max<In0Shape::W, In1Shape::W>::value>;
    static_assert(
        DerOutShape::N == OutShape::N,
        "Dimension N of output is not as expected from broadcast rules");
    static_assert(
        DerOutShape::C == OutShape::C,
        "Dimension C of output is not as expected from broadcast rules");
    static_assert(
        DerOutShape::H == OutShape::H,
        "Dimension H of output is not as expected from broadcast rules");
    static_assert(
        DerOutShape::W == OutShape::W,
        "Dimension W of output is not as expected from broadcast rules");
};

// Broadcast a unit operator. Follows NumPy-style broadcasting:
// https://numpy.org/doc/stable/user/basics.broadcasting.html
template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes,
          typename CompType>
struct Broadcast
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;
    using DataType = typename CompType::DataType;
    static const int NelemPerThread = CompType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutShape::W % NelemPerThread == 0,
                  "UnitOutShape::W must be divisible by NelemPerThread");

    // Conduct computation on input and broadcast the result to output.
    //
    // tn(int): index of the unit operator along the N dimension.
    // tc(int): index of the unit operator along the C dimension.
    // th(int): index of the unit operator along the H dimension.
    // tw(int): index of the unit operator along the W dimension.
    static DEVICE void run(DataType *out, const DataType *in0,
                           const DataType *in1, int tn, int tc, int th, int tw)
    {
        using InOutChk = BroadcastShapeChecker<In0Shape, In1Shape, OutShape>;

        for (int tid = UnitOp::thread_id();; tid += ThreadsNum) {
            int tid_w = (tid * NelemPerThread) % UnitOutShape::W;
            int tid_h =
                ((tid * NelemPerThread) / UnitOutShape::W) % UnitOutShape::H;
            int tid_c =
                ((tid * NelemPerThread) / UnitOutShape::HW) % UnitOutShape::C;
            int tid_n = (tid * NelemPerThread) / UnitOutShape::CHW;

            if (tid_n >= UnitOutShape::N) {
                break;
            }

            int idx_out = (tid_w + tw * UnitOutShape::W) +
                          (tid_h + th * UnitOutShape::H) * OutDims::W +
                          (tid_c + tc * UnitOutShape::C) * OutDims::HW +
                          (tid_n + tn * UnitOutShape::N) * OutDims::CHW;

            int idx_in0 =
                ((In0Shape::W == 1) ? 0 : (tid_w + tw * UnitOutShape::W)) +
                ((In0Shape::H == 1) ? 0 : (tid_h + th * UnitOutShape::H)) *
                    In0Dims::W +
                ((In0Shape::C == 1) ? 0 : (tid_c + tc * UnitOutShape::C)) *
                    In0Dims::HW +
                ((In0Shape::N == 1) ? 0 : (tid_n + tn * UnitOutShape::N)) *
                    In0Dims::CHW;

            int idx_in1 =
                ((In1Shape::W == 1) ? 0 : (tid_w + tw * UnitOutShape::W)) +
                ((In1Shape::H == 1) ? 0 : (tid_h + th * UnitOutShape::H)) *
                    In1Dims::W +
                ((In1Shape::C == 1) ? 0 : (tid_c + tc * UnitOutShape::C)) *
                    In1Dims::HW +
                ((In1Shape::N == 1) ? 0 : (tid_n + tn * UnitOutShape::N)) *
                    In1Dims::CHW;

            CompType::compute(&out[idx_out], &in0[idx_in0], &in1[idx_in1]);
        }
    }
};

} // namespace ark

#endif // ARK_KERNELS_BROADCAST_H_
