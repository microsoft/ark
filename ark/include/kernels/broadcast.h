#ifndef ARK_KERNELS_BROADCAST_H_
#define ARK_KERNELS_BROADCAST_H_

#include "static_math.h"
#include "unit_op.h"
#include "utils.h"

namespace ark {

// Static checker if InShape0 and InShape1 can be broadcasted into OutShape.
template <typename InShape0, typename InShape1, typename OutShape>
struct BroadcastShapeChecker
{
    static_assert(InShape0::N == 1 || InShape1::N == 1 ||
                      InShape0::N == InShape1::N,
                  "Cannot broadcast dimension N of inputs");
    static_assert(InShape0::C == 1 || InShape1::C == 1 ||
                      InShape0::C == InShape1::C,
                  "Cannot broadcast dimension C of inputs");
    static_assert(InShape0::H == 1 || InShape1::H == 1 ||
                      InShape0::H == InShape1::H,
                  "Cannot broadcast dimension H of inputs");
    static_assert(InShape0::W == 1 || InShape1::W == 1 ||
                      InShape0::W == InShape1::W,
                  "Cannot broadcast dimension W of inputs");

    // Derived OutShape.
    using DerOutShape = Vec<math::max<InShape0::N, InShape1::N>::value,
                            math::max<InShape0::C, InShape1::C>::value,
                            math::max<InShape0::H, InShape1::H>::value,
                            math::max<InShape0::W, InShape1::W>::value>;
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
          typename CompType, typename DataType, int NelemPerThread>
struct Broadcast
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutShape::W % NelemPerThread == 0,
                  "UnitOutShape::W must be divisible by NelemPerThread");

    // Conduct computation on input and broadcast the result to output.
    //
    // tn(int): index of the unit operator along the N dimension.
    // tc(int): index of the unit operator along the C dimension.
    // th(int): index of the unit operator along the H dimension.
    // tw(int): index of the unit operator along the W dimension.
    static DEVICE void run(DataType *out, DataType *in0, DataType *in1, int tn,
                           int tc, int th, int tw)
    {
        using InOutChk = BroadcastShapeChecker<In0Shape, In1Shape, OutShape>;

        for (int tid = UnitOp::thread_id();; tid += ThreadsNum) {
            int tid_w = (tid * NelemPerThread) % UnitOutShape::W;
            int tid_h =
                ((tid * NelemPerThread) / UnitOutShape::W) % UnitOutShape::H;
            int tid_c =
                ((tid * NelemPerThread) / UnitOutShape::W / UnitOutShape::H) %
                UnitOutShape::C;
            int tid_n =
                tid / UnitOutShape::W / UnitOutShape::H / UnitOutShape::C;

            if (tid_n >= UnitOutShape::N) {
                break;
            }

            int idx_out =
                (tid_w + tw * UnitOutShape::W) +
                (tid_h + th * UnitOutShape::H) * OutDims::W +
                (tid_c + tc * UnitOutShape::C) * OutDims::W * OutDims::H +
                (tid_n + tn * UnitOutShape::N) * OutDims::W * OutDims::H *
                    OutDims::C;

            int idx_in0 =
                ((In0Shape::W == 1) ? 0 : (tid_w + tw * UnitOutShape::W)) +
                ((In0Shape::H == 1) ? 0 : (tid_h + th * UnitOutShape::H)) *
                    In0Dims::W +
                ((In0Shape::C == 1) ? 0 : (tid_c + tc * UnitOutShape::C)) *
                    In0Dims::W * In0Dims::H +
                ((In0Shape::N == 1) ? 0 : (tid_n + tn * UnitOutShape::N)) *
                    In0Dims::W * In0Dims::H * In0Dims::C;

            int idx_in1 =
                ((In1Shape::W == 1) ? 0 : (tid_w + tw * UnitOutShape::W)) +
                ((In1Shape::H == 1) ? 0 : (tid_h + th * UnitOutShape::H)) *
                    In1Dims::W +
                ((In1Shape::C == 1) ? 0 : (tid_c + tc * UnitOutShape::C)) *
                    In1Dims::W * In1Dims::H +
                ((In1Shape::N == 1) ? 0 : (tid_n + tn * UnitOutShape::N)) *
                    In1Dims::W * In1Dims::H * In1Dims::C;

            CompType::compute<NelemPerThread>(&out[idx_out], &in0[idx_in0],
                                              &in1[idx_in1]);
        }
    }
};

} // namespace ark

#endif // ARK_KERNELS_BROADCAST_H_
