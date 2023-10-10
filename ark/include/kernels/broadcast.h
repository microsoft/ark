// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BROADCAST_H_
#define ARK_KERNELS_BROADCAST_H_

#include "common.h"

namespace ark {

template <typename _IntrinsicType, typename _InShape, typename _InputType,
          typename _OutputType, int _NelemPerThread>
struct Broadcast1Intrinsic {
    using InputType = _InputType;
    using OutputType = _OutputType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(OutputType *out, const InputType *in) {
        if (_InShape::W == 1) {
            *out = _IntrinsicType::compute(*in);
        } else {
#pragma unroll
            for (int i = 0; i < NelemPerThread; ++i) {
                out[i] = _IntrinsicType::compute(in[i]);
            }
        }
    }
};

template <typename _IntrinsicType, typename _InShape>
struct Broadcast1Intrinsic<_IntrinsicType, _InShape, float, float, 2> {
    using InputType = float;
    using OutputType = float;
    static const int NelemPerThread = 2;

    static DEVICE void compute(float *out, const float *in) {
        if (_InShape::W == 1) {
            *out = _IntrinsicType::compute(*in);
        } else {
            float2 *pout = (float2 *)out;
            float2 *pin = (float2 *)in;
            pout->x = _IntrinsicType::compute(pin->x);
            pout->y = _IntrinsicType::compute(pin->y);
        }
    }
};

template <typename _IntrinsicType, typename _InShape>
struct Broadcast1Intrinsic<_IntrinsicType, _InShape, float, float, 4> {
    using InputType = float;
    using OutputType = float;
    static const int NelemPerThread = 4;

    static DEVICE void compute(float *out, const float *in) {
        if (_InShape::W == 1) {
            *out = _IntrinsicType::compute(*in);
        } else {
            longlong2 reg_out;
            longlong2 reg_in;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_in.x), "=l"(reg_in.y)
                         : "l"(in)
                         : "memory");
            float4 *pout = (float4 *)&reg_out;
            float4 *pin = (float4 *)&reg_in;
            pout->w = _IntrinsicType::compute(pin->w);
            pout->x = _IntrinsicType::compute(pin->x);
            pout->y = _IntrinsicType::compute(pin->y);
            pout->z = _IntrinsicType::compute(pin->z);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(out), "l"(reg_out.x), "l"(reg_out.y)
                         : "memory");
        }
    }
};

template <typename _IntrinsicType, typename _InShape>
struct Broadcast1Intrinsic<_IntrinsicType, _InShape, half, half, 2> {
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *out, const half *in) {
        if (_InShape::W == 1) {
            *out = _IntrinsicType::compute(*in);
        } else {
            *(__half2 *)out = _IntrinsicType::compute(*(__half2 *)in);
        }
    }
};

template <typename _IntrinsicType, typename _InShape>
struct Broadcast1Intrinsic<_IntrinsicType, _InShape, half, half, 4> {
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 4;

    static DEVICE void compute(half *out, const half *in) {
        if (_InShape::W == 1) {
            *out = _IntrinsicType::compute(*in);
        } else {
            uint64_t reg_in = *(uint64_t *)in;
            uint64_t reg_out;
            __half2 *pin = (__half2 *)&reg_in;
            __half2 *pout = (__half2 *)&reg_out;
            pout[0] = _IntrinsicType::compute(pin[0]);
            pout[1] = _IntrinsicType::compute(pin[1]);
            *(uint64_t *)out = reg_out;
        }
    }
};

template <typename _IntrinsicType, typename _InShape>
struct Broadcast1Intrinsic<_IntrinsicType, _InShape, half, half, 8> {
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 8;

    static DEVICE void compute(half *out, const half *in) {
        if (_InShape::W == 1) {
            *out = _IntrinsicType::compute(*in);
        } else {
            longlong2 reg_in;
            longlong2 reg_out;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_in.x), "=l"(reg_in.y)
                         : "l"(in)
                         : "memory");
            __half2 *pin = (__half2 *)&reg_in;
            __half2 *pout = (__half2 *)&reg_out;
            pout[0] = _IntrinsicType::compute(pin[0]);
            pout[1] = _IntrinsicType::compute(pin[1]);
            pout[2] = _IntrinsicType::compute(pin[2]);
            pout[3] = _IntrinsicType::compute(pin[3]);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(out), "l"(reg_out.x), "l"(reg_out.y)
                         : "memory");
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
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            float4 *pb = (float4 *)&reg_b;
            float4 *pc = (float4 *)&reg_c;
            float v = *a;
            pc->w = _IntrinsicType::compute(v, pb->w);
            pc->x = _IntrinsicType::compute(v, pb->x);
            pc->y = _IntrinsicType::compute(v, pb->y);
            pc->z = _IntrinsicType::compute(v, pb->z);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else if (_In1Shape::W == 1) {
            longlong2 reg_a;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            float4 *pa = (float4 *)&reg_a;
            float4 *pc = (float4 *)&reg_c;
            float v = *b;
            pc->w = _IntrinsicType::compute(pa->w, v);
            pc->x = _IntrinsicType::compute(pa->x, v);
            pc->y = _IntrinsicType::compute(pa->y, v);
            pc->z = _IntrinsicType::compute(pa->z, v);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else {
            longlong2 reg_a;
            longlong2 reg_b;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            float4 *pa = (float4 *)&reg_a;
            float4 *pb = (float4 *)&reg_b;
            float4 *pc = (float4 *)&reg_c;
            pc->w = _IntrinsicType::compute(pa->w, pb->w);
            pc->x = _IntrinsicType::compute(pa->x, pb->x);
            pc->y = _IntrinsicType::compute(pa->y, pb->y);
            pc->z = _IntrinsicType::compute(pa->z, pb->z);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, half, half,
                           2> {
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *c, const half *a, const half *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            __half2 *pb = (__half2 *)b;
            *(__half2 *)c =
                _IntrinsicType::compute(__half2half2(*(const __half *)a), *pb);
        } else if (_In1Shape::W == 1) {
            __half2 *pa = (__half2 *)a;
            *(__half2 *)c =
                _IntrinsicType::compute(*pa, __half2half2(*(const __half *)b));
        } else {
            __half2 *pa = (__half2 *)a;
            __half2 *pb = (__half2 *)b;
            *(__half2 *)c = _IntrinsicType::compute(*pa, *pb);
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, half, half,
                           4> {
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 4;

    static DEVICE void compute(half *c, const half *a, const half *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            uint64_t reg_b = *(uint64_t *)b;
            uint64_t reg_c;
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)a);
            pc[0] = _IntrinsicType::compute(v, pb[0]);
            pc[1] = _IntrinsicType::compute(v, pb[1]);
            *(uint64_t *)c = reg_c;
        } else if (_In1Shape::W == 1) {
            uint64_t reg_a = *(uint64_t *)a;
            uint64_t reg_c;
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)b);
            pc[0] = _IntrinsicType::compute(pa[0], v);
            pc[1] = _IntrinsicType::compute(pa[1], v);
            *(uint64_t *)c = reg_c;
        } else {
            uint64_t reg_a = *(uint64_t *)a;
            uint64_t reg_b = *(uint64_t *)b;
            uint64_t reg_c;
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            pc[0] = _IntrinsicType::compute(pa[0], pb[0]);
            pc[1] = _IntrinsicType::compute(pa[1], pb[1]);
            *(uint64_t *)c = reg_c;
        }
    }
};

template <typename _IntrinsicType, typename _In0Shape, typename _In1Shape>
struct Broadcast2Intrinsic<_IntrinsicType, _In0Shape, _In1Shape, half, half,
                           8> {
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 8;

    static DEVICE void compute(half *c, const half *a, const half *b) {
        if (_In0Shape::W == 1 && _In1Shape::W == 1) {
            *c = _IntrinsicType::compute(*a, *b);
        } else if (_In0Shape::W == 1) {
            longlong2 reg_b;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)a);
            pc[0] = _IntrinsicType::compute(v, pb[0]);
            pc[1] = _IntrinsicType::compute(v, pb[1]);
            pc[2] = _IntrinsicType::compute(v, pb[2]);
            pc[3] = _IntrinsicType::compute(v, pb[3]);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else if (_In1Shape::W == 1) {
            longlong2 reg_a;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pc = (__half2 *)&reg_c;
            __half2 v = __half2half2(*(const __half *)b);
            pc[0] = _IntrinsicType::compute(pa[0], v);
            pc[1] = _IntrinsicType::compute(pa[1], v);
            pc[2] = _IntrinsicType::compute(pa[2], v);
            pc[3] = _IntrinsicType::compute(pa[3], v);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        } else {
            longlong2 reg_a;
            longlong2 reg_b;
            longlong2 reg_c;
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_a.x), "=l"(reg_a.y)
                         : "l"(a)
                         : "memory");
            asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
                         : "=l"(reg_b.x), "=l"(reg_b.y)
                         : "l"(b)
                         : "memory");
            __half2 *pa = (__half2 *)&reg_a;
            __half2 *pb = (__half2 *)&reg_b;
            __half2 *pc = (__half2 *)&reg_c;
            pc[0] = _IntrinsicType::compute(pa[0], pb[0]);
            pc[1] = _IntrinsicType::compute(pa[1], pb[1]);
            pc[2] = _IntrinsicType::compute(pa[2], pb[2]);
            pc[3] = _IntrinsicType::compute(pa[3], pb[3]);
            asm volatile("st.global.v2.u64 [%0], {%1,%2};"
                         :
                         : "l"(c), "l"(reg_c.x), "l"(reg_c.y)
                         : "memory");
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

// Static checker if InShape can be broadcasted into OutShape.
template <typename InShape, typename OutShape>
struct BroadcastShapeChecker1 {
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
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename Intrinsic>
struct Broadcast1 {
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;
    using InputType = typename Intrinsic::InputType;
    using OutputType = typename Intrinsic::OutputType;
    static const int NelemPerThread = Intrinsic::NelemPerThread;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

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

            Intrinsic::compute(&out[idx_out], &in[idx_in]);
        }
    }
};

// Broadcast a unit operator. Follows NumPy-style broadcasting:
// https://numpy.org/doc/stable/user/basics.broadcasting.html
template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes,
          typename Intrinsic>
struct Broadcast2 {
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;
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

            Intrinsic::compute(&out[idx_out], &in0[idx_in0], &in1[idx_in1]);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

// Broadcast2 with a default `NelemPerThread` and the intrinsic template.
template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes,
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
        NumThreads, SmemBytes,
        Broadcast2Intrinsic<IntrinsicType, In0Shape, In1Shape, In0DataType,
                            OutDataType, NelemPerThread>>::run(c, a, b,
                                                               uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_BROADCAST_H_
