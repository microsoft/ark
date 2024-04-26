// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BROADCAST_H_
#define ARK_KERNELS_BROADCAST_H_

#include "bf16.h"
#include "checker.h"
#include "fp16.h"
#include "fp32.h"
#include "integer.h"
#include "load_store.h"
#include "type_intrinsics.h"
#include "unit_op.h"
#include "vector_type.h"

namespace ark {

// NOTE: Broadcast does not automatically cast input to output type.
// Type casting should be handled by `IntrinsicType::compute()` if needed.

template <typename _IntrinsicType, typename _InShape, bool _IsHConsec,
          typename _InputType, typename _OutputType, int _NelemPerThread>
struct Broadcast1Intrinsic {
    using InputType = _InputType;
    using OutputType = _OutputType;
    static const int NelemPerThread = _NelemPerThread;
    static const bool BroadcastInput =
        (_IsHConsec && _InShape::H == 1) || (!_IsHConsec && _InShape::W == 1);

    static_assert(math::is_pow2<NelemPerThread>::value,
                  "NelemPerThread must be power of 2");
    static_assert(math::is_pow2<sizeof(InputType)>::value,
                  "InputType size must be power of 2");
    static_assert(math::is_pow2<sizeof(OutputType)>::value,
                  "OutputType size must be power of 2");
    static_assert(sizeof(InputType) <= 8,
                  "InputType size must be no larger than 8");
    static_assert(sizeof(OutputType) <= 8,
                  "OutputType size must be no larger than 8");

    static DEVICE void load(InputType *stage, const InputType *in) {
        if constexpr (BroadcastInput) {
            *stage = *in;
        } else {
            ark::load<NelemPerThread * sizeof(InputType)>(stage, in);
        }
    }

    static DEVICE void intrinsic(OutputType *result, const InputType *stage) {
        if constexpr (BroadcastInput) {
            constexpr int OutputVtypeSize =
                math::min<type::VtypeMaxSize<OutputType>::value,
                          NelemPerThread>::value;
            constexpr int NumComputeLoop = NelemPerThread / OutputVtypeSize;
            using OutputVtype =
                typename type::Vtype<OutputType, OutputVtypeSize>::type;

            OutputType reg = _IntrinsicType::compute(*stage);
#pragma unroll
            for (int i = 0; i < NumComputeLoop; ++i) {
                *(reinterpret_cast<OutputVtype *>(result) + i) =
                    type::Replicate::compute<OutputVtypeSize, OutputType>(reg);
            }
        } else {
            VectorCompute<NelemPerThread, _IntrinsicType>::compute(result,
                                                                   stage);
        }
    }

    static DEVICE void store(OutputType *out, const OutputType *result) {
        ark::store<NelemPerThread * sizeof(OutputType)>(out, result);
    }

    static DEVICE void compute(OutputType *out, const InputType *in) {
        if constexpr (BroadcastInput) {
            InputType stage;
            load(&stage, in);
            OutputType result[NelemPerThread];
            intrinsic(result, &stage);
            store(out, result);
        } else {
            InputType stage[NelemPerThread];
            load(stage, in);
            OutputType result[NelemPerThread];
            intrinsic(result, stage);
            store(out, result);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape,
          bool _IsHConsec, typename _InputType, typename _OutputType,
          int _NelemPerThread>
struct Broadcast2Intrinsic {
    using InputType = _InputType;
    using OutputType = _OutputType;
    static const int NelemPerThread = _NelemPerThread;
    static const bool BroadcastInput0 =
        (_IsHConsec && _In0Shape::H == 1) || (!_IsHConsec && _In0Shape::W == 1);
    static const bool BroadcastInput1 =
        (_IsHConsec && _In1Shape::H == 1) || (!_IsHConsec && _In1Shape::W == 1);

    static_assert(math::is_pow2<NelemPerThread>::value,
                  "NelemPerThread must be power of 2");
    static_assert(math::is_pow2<sizeof(InputType)>::value,
                  "InputType size must be power of 2");
    static_assert(math::is_pow2<sizeof(OutputType)>::value,
                  "OutputType size must be power of 2");
    static_assert(sizeof(InputType) <= 8,
                  "InputType size must be no larger than 8");
    static_assert(sizeof(OutputType) <= 8,
                  "OutputType size must be no larger than 8");

    static DEVICE void load0(InputType *stage0, const InputType *in0) {
        if constexpr (BroadcastInput0) {
            *stage0 = *in0;
        } else {
            ark::load<NelemPerThread * sizeof(InputType)>(stage0, in0);
        }
    }

    static DEVICE void load1(InputType *stage1, const InputType *in1) {
        if constexpr (BroadcastInput1) {
            *stage1 = *in1;
        } else {
            ark::load<NelemPerThread * sizeof(InputType)>(stage1, in1);
        }
    }

    static DEVICE void intrinsic(OutputType *result, const InputType *stage0,
                                 const InputType *stage1) {
        if constexpr (BroadcastInput0 && BroadcastInput1) {
            constexpr int OutputVtypeSize =
                math::min<type::VtypeMaxSize<OutputType>::value,
                          NelemPerThread>::value;
            constexpr int NumComputeLoop = NelemPerThread / OutputVtypeSize;
            using OutputVtype =
                typename type::Vtype<OutputType, OutputVtypeSize>::type;

            OutputType reg = _IntrinsicType::compute(*stage0, *stage1);
#pragma unroll
            for (int i = 0; i < NumComputeLoop; ++i) {
                *(reinterpret_cast<OutputVtype *>(result) + i) =
                    type::Replicate::compute<OutputVtypeSize, OutputType>(reg);
            }
        } else if constexpr (BroadcastInput0 || BroadcastInput1) {
            constexpr int TmpMin =
                math::min<type::VtypeMaxSize<OutputType>::value,
                          NelemPerThread>::value;
            constexpr int VtypeSize = math::min<
                TmpMin, IntrinsicCompute2VtypeMaxSize<_IntrinsicType, InputType,
                                                      TmpMin>::value>::value;
            using OutputVtype =
                typename type::Vtype<OutputType, VtypeSize>::type;
            using InputVtype = typename type::Vtype<InputType, VtypeSize>::type;
            constexpr int NumVtype = NelemPerThread / VtypeSize;

            static_assert(NelemPerThread % VtypeSize == 0,
                          "NelemPerThread must be divisible by VtypeSize");
            OutputVtype *out_vtype = reinterpret_cast<OutputVtype *>(result);
            if constexpr (BroadcastInput0) {
                const InputVtype in0_vtype =
                    type::Replicate::compute<VtypeSize, InputType>(*stage0);
                const InputVtype *in1_vtype =
                    reinterpret_cast<const InputVtype *>(stage1);
#pragma unroll
                for (int i = 0; i < NumVtype; ++i) {
                    out_vtype[i] =
                        _IntrinsicType::compute(in0_vtype, in1_vtype[i]);
                }
            } else {
                const InputVtype *in0_vtype =
                    reinterpret_cast<const InputVtype *>(stage0);
                const InputVtype in1_vtype =
                    type::Replicate::compute<VtypeSize, InputType>(*stage1);
#pragma unroll
                for (int i = 0; i < NumVtype; ++i) {
                    out_vtype[i] =
                        _IntrinsicType::compute(in0_vtype[i], in1_vtype);
                }
            }
        } else {
            VectorCompute<NelemPerThread, _IntrinsicType>::compute(
                result, stage0, stage1);
        }
    }

    static DEVICE void store(OutputType *out, const OutputType *result) {
        ark::store<NelemPerThread * sizeof(OutputType)>(out, result);
    }

    static DEVICE void compute(OutputType *out, const InputType *in0,
                               const InputType *in1) {
        if constexpr (BroadcastInput0 && BroadcastInput1) {
            InputType stage0;
            InputType stage1;
            load0(&stage0, in0);
            load1(&stage1, in1);
            OutputType result[NelemPerThread];
            intrinsic(result, &stage0, &stage1);
            store(out, result);
        } else if constexpr (BroadcastInput0) {
            InputType stage0;
            InputType stage1[NelemPerThread];
            load0(&stage0, in0);
            load1(stage1, in1);
            OutputType result[NelemPerThread];
            intrinsic(result, &stage0, stage1);
            store(out, result);
        } else if constexpr (BroadcastInput1) {
            InputType stage0[NelemPerThread];
            InputType stage1;
            load0(stage0, in0);
            load1(&stage1, in1);
            OutputType result[NelemPerThread];
            intrinsic(result, stage0, &stage1);
            store(out, result);
        } else {
            InputType stage0[NelemPerThread];
            InputType stage1[NelemPerThread];
            load0(stage0, in0);
            load1(stage1, in1);
            OutputType result[NelemPerThread];
            intrinsic(result, stage0, stage1);
            store(out, result);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

// Static checker if InShape can be broadcasted into OutShape.
template <typename InShape, typename OutShape>
struct BroadcastShapeChecker1 {
    static_assert(InShape::N == 1 || InShape::N == OutShape::N,
                  "Cannot broadcast dimension N of the input");
    static_assert(InShape::C == 1 || InShape::C == OutShape::C,
                  "Cannot broadcast dimension C of the input");
    static_assert(InShape::H == 1 || InShape::H == OutShape::H,
                  "Cannot broadcast dimension H of the input");
    static_assert(InShape::W == 1 || InShape::W == OutShape::W,
                  "Cannot broadcast dimension W of the input");

    // Derived OutShape.
    using DerOutShape = OutShape;
};

// Static checker if In0Shape and In1Shape can be broadcasted into OutShape.
template <typename In0Shape, typename In1Shape, typename OutShape>
struct BroadcastShapeChecker2 {
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
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename Intrinsic>
struct Broadcast1 {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;
    using InputType = typename Intrinsic::InputType;
    using OutputType = typename Intrinsic::OutputType;
    static constexpr int NelemPerThread = Intrinsic::NelemPerThread;
    static constexpr bool IsHConsec = (OutDims::W == 1 && UnitOutDims::W == 1);
    static constexpr int ConsecutiveDimLen =
        IsHConsec ? UnitOutDims::H : UnitOutDims::W;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(ConsecutiveDimLen % NelemPerThread == 0,
                  "ConsecutiveDimLen must be divisible by NelemPerThread");

    /// Conduct computation on one input and broadcast the result to output.
    /// @param out Output data.
    /// @param in1 Input data.
    /// @param uop_idx Index of the unit operator.
    static DEVICE void run(OutputType *out, const InputType *in, int uop_idx) {
        using InOutChk = BroadcastShapeChecker1<InShape, OutShape>;

        int un = UnitOp::uop_idx_n(uop_idx);
        int uc = UnitOp::uop_idx_c(uop_idx);
        int uh = UnitOp::uop_idx_h(uop_idx);
        int uw = UnitOp::uop_idx_w(uop_idx);

        for (int tid = UnitOp::thread_id();; tid += UnitOp::NumThreads) {
            int tid_n = (tid * NelemPerThread) / UnitOutDims::CHW;
            int tid_c =
                ((tid * NelemPerThread) / UnitOutDims::HW) % UnitOutDims::C;
            int tid_h;
            int tid_w;
            if constexpr (IsHConsec) {
                tid_h = (tid * NelemPerThread) % UnitOutDims::H;
                tid_w = 0;
            } else {
                tid_h =
                    ((tid * NelemPerThread) / UnitOutDims::W) % UnitOutDims::H;
                tid_w = (tid * NelemPerThread) % UnitOutDims::W;
            }

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

            Intrinsic::compute(&out[idx_out], &in[idx_in]);
        }

        UnitOp::sync_threads();
    }
};

// Broadcast a unit operator. Follows NumPy-style broadcasting:
// https://numpy.org/doc/stable/user/basics.broadcasting.html
template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes, typename Intrinsic>
struct Broadcast2 {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;
    using InputType = typename Intrinsic::InputType;
    using OutputType = typename Intrinsic::OutputType;
    static constexpr int NelemPerThread = Intrinsic::NelemPerThread;
    static constexpr bool IsHConsec = (OutDims::W == 1 && UnitOutDims::W == 1);
    static constexpr int ConsecutiveDimLen =
        IsHConsec ? UnitOutDims::H : UnitOutDims::W;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(ConsecutiveDimLen % NelemPerThread == 0,
                  "ConsecutiveDimLen must be divisible by NelemPerThread");

    /// Conduct computation on two inputs and broadcast the result to output.
    /// @param out Output data.
    /// @param in0 Input data 0.
    /// @param in1 Input data 1.
    /// @param uop_idx Index of the unit operator.
    static DEVICE void run(OutputType *out, const InputType *in0,
                           const InputType *in1, int uop_idx) {
        using InOutChk = BroadcastShapeChecker2<In0Shape, In1Shape, OutShape>;

        int un = UnitOp::uop_idx_n(uop_idx);
        int uc = UnitOp::uop_idx_c(uop_idx);
        int uh = UnitOp::uop_idx_h(uop_idx);
        int uw = UnitOp::uop_idx_w(uop_idx);

        for (int tid = UnitOp::thread_id();; tid += UnitOp::NumThreads) {
            int tid_n = (tid * NelemPerThread) / UnitOutDims::CHW;
            int tid_c =
                ((tid * NelemPerThread) / UnitOutDims::HW) % UnitOutDims::C;
            int tid_h;
            int tid_w;
            if constexpr (IsHConsec) {
                tid_h = (tid * NelemPerThread) % UnitOutDims::H;
                tid_w = 0;
            } else {
                tid_h =
                    ((tid * NelemPerThread) / UnitOutDims::W) % UnitOutDims::H;
                tid_w = (tid * NelemPerThread) % UnitOutDims::W;
            }

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

            Intrinsic::compute(&out[idx_out], &in0[idx_in0], &in1[idx_in1]);
        }

        UnitOp::sync_threads();
    }
};

////////////////////////////////////////////////////////////////////////////////

template <typename InDims, typename InShape, typename InDataType,
          typename OutDims, typename OutShape, typename OutDataType,
          typename IntrinsicType, typename UnitOutDims, int NumWarps,
          int SmemBytes>
struct DefaultBroadcast1
    : public Broadcast1<
          InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes,
          Broadcast1Intrinsic<IntrinsicType, InShape,
                              (OutDims::W == 1 && UnitOutDims::W == 1),
                              InDataType, OutDataType,
                              DefaultNelemPerThread<OutDims, OutDataType,
                                                    UnitOutDims>::value>> {};

////////////////////////////////////////////////////////////////////////////////

// Broadcast2 with a default `NelemPerThread` and the intrinsic template.
template <typename In0Dims, typename In0Shape, typename In0DataType,
          typename In1Dims, typename In1Shape, typename In1DataType,
          typename OutDims, typename OutShape, typename OutDataType,
          typename IntrinsicType, typename UnitOutDims, int NumWarps,
          int SmemBytes>
struct DefaultBroadcast2
    : public Broadcast2<
          In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape, UnitOutDims,
          NumWarps, SmemBytes,
          Broadcast2Intrinsic<IntrinsicType, In0Shape, In1Shape,
                              (OutDims::W == 1 && UnitOutDims::W == 1),
                              In0DataType, OutDataType,
                              DefaultNelemPerThread<OutDims, OutDataType,
                                                    UnitOutDims>::value>> {};

}  // namespace ark

#endif  // ARK_KERNELS_BROADCAST_H_
