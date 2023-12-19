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
          typename _InputType, typename _OutputType, int _NelemPerThread>
struct Broadcast2Intrinsic {
    using InputType = _InputType;
    using OutputType = _OutputType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(OutputType *c, const InputType *a,
                               const InputType *b) {
        *c = _IntrinsicType::compute(*a, *b);
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            // do nothing
        } else if (_In0Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = _IntrinsicType::compute(*a, b[i]);
            }
        } else if (_In1Shape::W == 1) {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = _IntrinsicType::compute(a[i], *b);
            }
        } else {
#pragma unroll
            for (int i = 1; i < NelemPerThread; ++i) {
                c[i] = _IntrinsicType::compute(a[i], b[i]);
            }
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, float, float,
                           2> {
    using InputType = float;
    using OutputType = float;
    static const int NelemPerThread = 2;

    static DEVICE void compute(float *c, const float *a, const float *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            float2 *pb = (float2 *)b;
            float2 *pc = (float2 *)c;
            pc->x = _IntrinsicType::compute(*a, pb->x);
            pc->y = _IntrinsicType::compute(*a, pb->y);
        } else if (_In1Shape::W == 1) {
            float2 *pa = (float2 *)a;
            float2 *pc = (float2 *)c;
            pc->x = _IntrinsicType::compute(pa->x, *b);
            pc->y = _IntrinsicType::compute(pa->y, *b);
        } else {
            float2 *pa = (float2 *)a;
            float2 *pb = (float2 *)b;
            float2 *pc = (float2 *)c;
            pc->x = _IntrinsicType::compute(pa->x, pb->x);
            pc->y = _IntrinsicType::compute(pa->y, pb->y);
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, float, float,
                           4> {
    using InputType = float;
    using OutputType = float;
    static const int NelemPerThread = 4;

    static DEVICE void compute(float *c, const float *a, const float *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            longlong2 reg_b;
            load<16>(&reg_b, b);
            longlong2 reg_c;
            float4 *pb = (float4 *)&reg_b;
            float4 *pc = (float4 *)&reg_c;
            float v = *a;
            pc->w = _IntrinsicType::compute(v, pb->w);
            pc->x = _IntrinsicType::compute(v, pb->x);
            pc->y = _IntrinsicType::compute(v, pb->y);
            pc->z = _IntrinsicType::compute(v, pb->z);
            store<16>(c, &reg_c);
        } else if (_In1Shape::W == 1) {
            longlong2 reg_a;
            load<16>(&reg_a, a);
            longlong2 reg_c;
            float4 *pa = (float4 *)&reg_a;
            float4 *pc = (float4 *)&reg_c;
            float v = *b;
            pc->w = _IntrinsicType::compute(pa->w, v);
            pc->x = _IntrinsicType::compute(pa->x, v);
            pc->y = _IntrinsicType::compute(pa->y, v);
            pc->z = _IntrinsicType::compute(pa->z, v);
            store<16>(c, &reg_c);
        } else {
            longlong2 reg_a;
            longlong2 reg_b;
            load<16>(&reg_a, a);
            load<16>(&reg_b, b);
            longlong2 reg_c;
            float4 *pa = (float4 *)&reg_a;
            float4 *pb = (float4 *)&reg_b;
            float4 *pc = (float4 *)&reg_c;
            pc->w = _IntrinsicType::compute(pa->w, pb->w);
            pc->x = _IntrinsicType::compute(pa->x, pb->x);
            pc->y = _IntrinsicType::compute(pa->y, pb->y);
            pc->z = _IntrinsicType::compute(pa->z, pb->z);
            store<16>(c, &reg_c);
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, fp16, fp16,
                           2> {
    using InputType = fp16;
    using OutputType = fp16;
    static const int NelemPerThread = 2;

    static DEVICE void compute(fp16 *c, const fp16 *a, const fp16 *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            fp16x2 *pb = (fp16x2 *)b;
            *(fp16x2 *)c = _IntrinsicType::compute(__half2half2(*a), *pb);
        } else if (_In1Shape::W == 1) {
            fp16x2 *pa = (fp16x2 *)a;
            *(fp16x2 *)c = _IntrinsicType::compute(*pa, __half2half2(*b));
        } else {
            fp16x2 *pa = (fp16x2 *)a;
            fp16x2 *pb = (fp16x2 *)b;
            *(fp16x2 *)c = _IntrinsicType::compute(*pa, *pb);
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, fp16, fp16,
                           4> {
    using InputType = fp16;
    using OutputType = fp16;
    static const int NelemPerThread = 4;

    static DEVICE void compute(fp16 *c, const fp16 *a, const fp16 *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            uint64_t reg_b = *(uint64_t *)b;
            uint64_t reg_c;
            fp16x2 *pb = (fp16x2 *)&reg_b;
            fp16x2 *pc = (fp16x2 *)&reg_c;
            fp16x2 v = __half2half2(*a);
            pc[0] = _IntrinsicType::compute(v, pb[0]);
            pc[1] = _IntrinsicType::compute(v, pb[1]);
            *(uint64_t *)c = reg_c;
        } else if (_In1Shape::W == 1) {
            uint64_t reg_a = *(uint64_t *)a;
            uint64_t reg_c;
            fp16x2 *pa = (fp16x2 *)&reg_a;
            fp16x2 *pc = (fp16x2 *)&reg_c;
            fp16x2 v = __half2half2(*b);
            pc[0] = _IntrinsicType::compute(pa[0], v);
            pc[1] = _IntrinsicType::compute(pa[1], v);
            *(uint64_t *)c = reg_c;
        } else {
            uint64_t reg_a = *(uint64_t *)a;
            uint64_t reg_b = *(uint64_t *)b;
            uint64_t reg_c;
            fp16x2 *pa = (fp16x2 *)&reg_a;
            fp16x2 *pb = (fp16x2 *)&reg_b;
            fp16x2 *pc = (fp16x2 *)&reg_c;
            pc[0] = _IntrinsicType::compute(pa[0], pb[0]);
            pc[1] = _IntrinsicType::compute(pa[1], pb[1]);
            *(uint64_t *)c = reg_c;
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, fp16, fp16,
                           8> {
    using InputType = fp16;
    using OutputType = fp16;
    static const int NelemPerThread = 8;

    static DEVICE void compute(fp16 *c, const fp16 *a, const fp16 *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            longlong2 reg_b;
            load<16>(&reg_b, b);
            longlong2 reg_c;
            fp16x2 *pb = (fp16x2 *)&reg_b;
            fp16x2 *pc = (fp16x2 *)&reg_c;
            fp16x2 v = __half2half2(*a);
            pc[0] = _IntrinsicType::compute(v, pb[0]);
            pc[1] = _IntrinsicType::compute(v, pb[1]);
            pc[2] = _IntrinsicType::compute(v, pb[2]);
            pc[3] = _IntrinsicType::compute(v, pb[3]);
            store<16>(c, &reg_c);
        } else if (_In1Shape::W == 1) {
            longlong2 reg_a;
            load<16>(&reg_a, a);
            longlong2 reg_c;
            fp16x2 *pa = (fp16x2 *)&reg_a;
            fp16x2 *pc = (fp16x2 *)&reg_c;
            fp16x2 v = __half2half2(*b);
            pc[0] = _IntrinsicType::compute(pa[0], v);
            pc[1] = _IntrinsicType::compute(pa[1], v);
            pc[2] = _IntrinsicType::compute(pa[2], v);
            pc[3] = _IntrinsicType::compute(pa[3], v);
            store<16>(c, &reg_c);
        } else {
            longlong2 reg_a;
            longlong2 reg_b;
            load<16>(&reg_a, a);
            load<16>(&reg_b, b);
            longlong2 reg_c;
            fp16x2 *pa = (fp16x2 *)&reg_a;
            fp16x2 *pb = (fp16x2 *)&reg_b;
            fp16x2 *pc = (fp16x2 *)&reg_c;
            pc[0] = _IntrinsicType::compute(pa[0], pb[0]);
            pc[1] = _IntrinsicType::compute(pa[1], pb[1]);
            pc[2] = _IntrinsicType::compute(pa[2], pb[2]);
            pc[3] = _IntrinsicType::compute(pa[3], pb[3]);
            store<16>(c, &reg_c);
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
    static const int NelemPerThread = Intrinsic::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

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

            Intrinsic::compute(&out[idx_out], &in0[idx_in0], &in1[idx_in1]);
        }

        UnitOp::sync_threads();
    }
};

////////////////////////////////////////////////////////////////////////////////

template <typename OutDims, typename OutDataType, typename UnitOutDims>
struct DefaultNelemPerThread {
    static constexpr int ConsecutiveDimLen =
        (OutDims::W == 1 && UnitOutDims::W == 1) ? UnitOutDims::H
                                                 : UnitOutDims::W;

    static const int value =
        (sizeof(OutDataType) <= 2 && ConsecutiveDimLen % 8 == 0)
            ? 8
            : (ConsecutiveDimLen % 4 == 0)
                  ? 4
                  : (ConsecutiveDimLen % 2 == 0) ? 2 : 1;
};

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
template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType,
          typename IntrinsicType>
DEVICE void broadcast2(OutDataType *c, const In0DataType *a,
                       const In1DataType *b, int uop_idx, int) {
    constexpr int NelemPerThread =
        (sizeof(OutDataType) <= 2 && UnitOutDims::W % 8 == 0)
            ? 8
            : (UnitOutDims::W % 4 == 0) ? 4 : (UnitOutDims::W % 2 == 0) ? 2 : 1;
    Broadcast2<
        In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape, UnitOutDims,
        NumWarps, SmemBytes,
        Broadcast2Intrinsic<IntrinsicType, In0Shape, In1Shape, In0DataType,
                            OutDataType, NelemPerThread>>::run(c, a, b,
                                                               uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_BROADCAST_H_
