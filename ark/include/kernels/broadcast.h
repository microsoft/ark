// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BROADCAST_H_
#define ARK_KERNELS_BROADCAST_H_

#include "common.h"

namespace ark {

// Static checker if InShape can be broadcasted into OutShape.
template <typename InShape, typename OutShape> struct BroadcastShapeChecker1
{
    static_assert(InShape::N == 1 || OutShape::N == 1 ||
                      InShape::N == OutShape::N,
                  "Cannot broadcast dimension N of the input");
    static_assert(InShape::C == 1 || OutShape::C == 1 ||
                      InShape::C == OutShape::C,
                  "Cannot broadcast dimension C of the input");
    static_assert(InShape::H == 1 || OutShape::H == 1 ||
                      InShape::H == OutShape::H,
                  "Cannot broadcast dimension H of the input");
    static_assert(InShape::W == 1 || OutShape::W == 1 ||
                      InShape::W == OutShape::W,
                  "Cannot broadcast dimension W of the input");

    // Derived OutShape.
    using DerOutShape = OutShape;
};

// Static checker if In0Shape and In1Shape can be broadcasted into OutShape.
template <typename In0Shape, typename In1Shape, typename OutShape>
struct BroadcastShapeChecker2
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
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename CompType>
struct Broadcast1
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;
    using InputType = typename CompType::InputType;
    using OutputType = typename CompType::OutputType;
    static const int NelemPerThread = CompType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

    /// Conduct computation on one input and broadcast the result to output.
    /// @param out Output data.
    /// @param in1 Input data.
    /// @param uop_idx Index of the unit operator.
    static DEVICE void run(OutputType *out, const InputType *in, int uop_idx)
    {
        using InOutChk = BroadcastShapeChecker1<InShape, OutShape>;

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

            int idx_out = (tid_w + uw * UnitOutDims::W) +
                          (tid_h + uh * UnitOutDims::H) * OutDims::W +
                          (tid_c + uc * UnitOutDims::C) * OutDims::HW +
                          (tid_n + un * UnitOutDims::N) * OutDims::CHW;

            int idx_in;

            if constexpr (VecIsEq<InShape, OutShape>::value &&
                          VecIsEq<InDims, OutDims>::value) {
                idx_in = idx_out;
            } else {
                idx_in =
                    ((InShape::W == 1) ? 0 : (tid_w + uw * UnitOutDims::W)) +
                    ((InShape::H == 1) ? 0 : (tid_h + uh * UnitOutDims::H)) *
                        InDims::W +
                    ((InShape::C == 1) ? 0 : (tid_c + uc * UnitOutDims::C)) *
                        InDims::HW +
                    ((InShape::N == 1) ? 0 : (tid_n + un * UnitOutDims::N)) *
                        InDims::CHW;
            }

            CompType::compute(&out[idx_out], &in[idx_in]);
        }
    }
};

// Broadcast a unit operator. Follows NumPy-style broadcasting:
// https://numpy.org/doc/stable/user/basics.broadcasting.html
template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes,
          typename CompType>
struct Broadcast2
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;
    using InputType = typename CompType::InputType;
    using OutputType = typename CompType::OutputType;
    static const int NelemPerThread = CompType::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

    /// Conduct computation on two inputs and broadcast the result to output.
    /// @param out Output data.
    /// @param in0 Input data 0.
    /// @param in1 Input data 1.
    /// @param uop_idx Index of the unit operator.
    static DEVICE void run(OutputType *out, const InputType *in0,
                           const InputType *in1, int uop_idx)
    {
        using InOutChk = BroadcastShapeChecker2<In0Shape, In1Shape, OutShape>;

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

            int idx_out = (tid_w + uw * UnitOutDims::W) +
                          (tid_h + uh * UnitOutDims::H) * OutDims::W +
                          (tid_c + uc * UnitOutDims::C) * OutDims::HW +
                          (tid_n + un * UnitOutDims::N) * OutDims::CHW;

            int idx_in0;
            int idx_in1;

            if constexpr (VecIsEq<In0Shape, OutShape>::value &&
                          VecIsEq<In0Dims, OutDims>::value) {
                idx_in0 = idx_out;
            } else {
                idx_in0 =
                    ((In0Shape::W == 1) ? 0 : (tid_w + uw * UnitOutDims::W)) +
                    ((In0Shape::H == 1) ? 0 : (tid_h + uh * UnitOutDims::H)) *
                        In0Dims::W +
                    ((In0Shape::C == 1) ? 0 : (tid_c + uc * UnitOutDims::C)) *
                        In0Dims::HW +
                    ((In0Shape::N == 1) ? 0 : (tid_n + un * UnitOutDims::N)) *
                        In0Dims::CHW;
            }

            if constexpr (VecIsEq<In1Shape, OutShape>::value &&
                          VecIsEq<In1Dims, OutDims>::value) {
                idx_in1 = idx_out;
            } else if constexpr (VecIsEq<In1Shape, In0Shape>::value &&
                                 VecIsEq<In1Dims, In0Dims>::value) {
                idx_in1 = idx_in0;
            } else {
                idx_in1 =
                    ((In1Shape::W == 1) ? 0 : (tid_w + uw * UnitOutDims::W)) +
                    ((In1Shape::H == 1) ? 0 : (tid_h + uh * UnitOutDims::H)) *
                        In1Dims::W +
                    ((In1Shape::C == 1) ? 0 : (tid_c + uc * UnitOutDims::C)) *
                        In1Dims::HW +
                    ((In1Shape::N == 1) ? 0 : (tid_n + un * UnitOutDims::N)) *
                        In1Dims::CHW;
            }

            CompType::compute(&out[idx_out], &in0[idx_in0], &in1[idx_in1]);
        }
    }
};

} // namespace ark

#endif // ARK_KERNELS_BROADCAST_H_
