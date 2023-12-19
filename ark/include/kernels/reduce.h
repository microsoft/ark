// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_REDUCE_H_
#define ARK_KERNELS_REDUCE_H_

#include "common/ewise.h"
#include "common/shfl.h"
#include "common/type_intrinsics.h"
#include "common/vector_type.h"

namespace ark {

typedef enum {
    N = 0,
    C = 1,
    H = 2,
    W = 3,
} AxisType;

// Shared memory for reduction.
template <typename DataType>
struct ReduceSharedStorage {
    DataType storage[Arch::ThreadsPerWarp];
};

// Reduce single-precision `val` within a single warp.
template <typename ReduceType, int LanesNum, typename DataType>
DEVICE DataType warpReduce(DataType val) {
    DataType res = val;
    DataType tmp;
    constexpr int iter =
        math::log2_up<math::min<LanesNum, Arch::ThreadsPerWarp>::value>::value;
    if constexpr (iter > 0) {
#pragma unroll
        for (int i = (1 << (iter - 1)); i > 0; i /= 2) {
            tmp = SHFL_XOR(res, i, i * 2);
            ReduceType::template reduce<1>(&res, &res, &tmp);
        }
    }
    return res;
}

// Reduce bf16 `val` within a single warp.
template <typename ReduceType, int LanesNum>
DEVICE bf16 warpReduce(bf16 val) {
    float tmp = type::Cast::compute<float>(val);
    tmp = warpReduce<ReduceType, LanesNum, float>(tmp);
    return type::Cast::compute<bf16>(tmp);
}

// Reduce single-precision `val` within multiple warps.
template <typename ReduceType, typename UnitOp, int LanesNum, typename DataType>
DEVICE DataType warpsReduce(DataType val, int tid, int smem_per_warp) {
    val = warpReduce<ReduceType, LanesNum>(val);
    if (LanesNum > Arch::ThreadsPerWarp) {
        ReduceSharedStorage<DataType> *shared =
            UnitOp::template shared_memory<ReduceSharedStorage<DataType>>(
                smem_per_warp);
        int laneId = tid & (Arch::ThreadsPerWarp - 1);
        int warpId = tid >> math::log2_up<Arch::ThreadsPerWarp>::value;
        if (laneId == 0) {
            shared->storage[warpId] = val;
        }
        UnitOp::sync_threads();
        if (laneId < (LanesNum >> math::log2_up<Arch::ThreadsPerWarp>::value)) {
            val = shared->storage[laneId];
        } else {
            ReduceType::template identity<1>(&val);
        }
        val = warpReduce<ReduceType, Arch::ThreadsPerWarp>(val);
    }
    return val;
}

// Check if InShape can be reduced into OutShape and if UnitOutDims is valid.
template <typename InShape, typename OutShape, typename UnitOutDims, int Axis>
struct ReduceShapeChecker {
    static_assert((InShape::N == OutShape::N) ||
                      (Axis == AxisType::N && OutShape::N == 1),
                  "Invalid dimension N");
    static_assert((InShape::C == OutShape::C) ||
                      (Axis == AxisType::C && OutShape::C == 1),
                  "Invalid dimension C");
    static_assert((InShape::H == OutShape::H) ||
                      (Axis == AxisType::H && OutShape::H == 1),
                  "Invalid dimension H");
    static_assert((InShape::W == OutShape::W) ||
                      (Axis == AxisType::W && OutShape::W == 1),
                  "Invalid dimension W");
    static_assert((UnitOutDims::N == 1) || (Axis != AxisType::N),
                  "Invalid UnitOutDims::N");
    static_assert((UnitOutDims::C == 1) || (Axis != AxisType::C),
                  "Invalid UnitOutDims::C");
    static_assert((UnitOutDims::H == 1) || (Axis != AxisType::H),
                  "Invalid UnitOutDims::H");
    static_assert((UnitOutDims::W == 1) || (Axis != AxisType::W),
                  "Invalid UnitOutDims::W");
};

struct ReduceTypeSum {
    template <int NelemPerThread, typename DataType>
    static DEVICE void identity(DataType *v) {
        if constexpr ((NelemPerThread >= 4) && (NelemPerThread % 4 == 0) &&
                      type::VtypeExists<DataType, 4>::value) {
            *reinterpret_cast<typename type::Vtype<DataType, 4>::type *>(v) =
                type::Constant<typename type::Vtype<DataType, 4>::type>::zero();
        } else if constexpr ((NelemPerThread == 2) &&
                             type::VtypeExists<DataType, 2>::value) {
            *reinterpret_cast<typename type::Vtype<DataType, 2>::type *>(v) =
                type::Constant<typename type::Vtype<DataType, 2>::type>::zero();
        } else {
#pragma unroll
            for (int elem = 0; elem < NelemPerThread; ++elem) {
                v[elem] = type::Constant<DataType>::zero();
            }
        }
    }

    template <int NelemPerThread, typename DataType>
    static DEVICE void reduce(DataType *out, const DataType *in0,
                              const DataType *in1) {
        VectorCompute<NelemPerThread, type::Add>::compute(out, in0, in1);
    }

    template <int NelemPerThread, typename DataType>
    static DEVICE void postReduce(DataType *out, const DataType *in, int) {
        VectorCompute<NelemPerThread, type::Identity>::compute(out, in);
    }
};

struct ReduceTypeMax {
    template <int NelemPerThread, typename DataType>
    static DEVICE void identity(DataType *v) {
        if constexpr ((NelemPerThread >= 4) && (NelemPerThread % 4 == 0) &&
                      type::VtypeExists<DataType, 4>::value) {
            *reinterpret_cast<typename type::Vtype<DataType, 4>::type *>(v) =
                type::Constant<
                    typename type::Vtype<DataType, 4>::type>::lowest();
        } else if constexpr ((NelemPerThread == 2) &&
                             type::VtypeExists<DataType, 2>::value) {
            *reinterpret_cast<typename type::Vtype<DataType, 2>::type *>(v) =
                type::Constant<
                    typename type::Vtype<DataType, 2>::type>::lowest();
        } else {
#pragma unroll
            for (int elem = 0; elem < NelemPerThread; ++elem) {
                v[elem] = type::Constant<DataType>::lowest();
            }
        }
    }

    template <int NelemPerThread, typename DataType>
    static DEVICE void reduce(DataType *out, const DataType *in0,
                              const DataType *in1) {
        VectorCompute<NelemPerThread, type::Max>::compute(out, in0, in1);
    }

    template <int NelemPerThread, typename DataType>
    static DEVICE void postReduce(DataType *out, const DataType *in,
                                  int nelem = 1) {
        VectorCompute<NelemPerThread, type::Identity>::compute(out, in);
    }
};

struct ReduceTypeMean {
    template <int NelemPerThread, typename DataType>
    static DEVICE void identity(DataType *v) {
        if constexpr ((NelemPerThread >= 4) && (NelemPerThread % 4 == 0) &&
                      type::VtypeExists<DataType, 4>::value) {
            *reinterpret_cast<typename type::Vtype<DataType, 4>::type *>(v) =
                type::Constant<typename type::Vtype<DataType, 4>::type>::zero();
        } else if constexpr ((NelemPerThread == 2) &&
                             type::VtypeExists<DataType, 2>::value) {
            *reinterpret_cast<typename type::Vtype<DataType, 2>::type *>(v) =
                type::Constant<typename type::Vtype<DataType, 2>::type>::zero();
        } else {
#pragma unroll
            for (int elem = 0; elem < NelemPerThread; ++elem) {
                v[elem] = type::Constant<DataType>::zero();
            }
        }
    }

    template <int NelemPerThread, typename DataType>
    static DEVICE void reduce(DataType *out, const DataType *in0,
                              const DataType *in1) {
        VectorCompute<NelemPerThread, type::Add>::compute(out, in0, in1);
    }

    template <int NelemPerThread, typename DataType>
    static DEVICE void postReduce(DataType *out, const DataType *in,
                                  int nelem = 1) {
        if constexpr ((NelemPerThread >= 4) && (NelemPerThread % 4 == 0) &&
                      type::VtypeExists<DataType, 4>::value) {
            int4 nelem4 = make_int4(nelem, nelem, nelem, nelem);
            *reinterpret_cast<typename type::Vtype<DataType, 4>::type *>(out) =
                type::Div::compute(
                    *reinterpret_cast<
                        const typename type::Vtype<DataType, 4>::type *>(in),
                    nelem4);
        } else if constexpr ((NelemPerThread == 2) &&
                             type::VtypeExists<DataType, 2>::value) {
            int2 nelem2 = make_int2(nelem, nelem);
            *reinterpret_cast<typename type::Vtype<DataType, 2>::type *>(out) =
                type::Div::compute(
                    *reinterpret_cast<
                        const typename type::Vtype<DataType, 2>::type *>(in),
                    nelem2);
        } else {
#pragma unroll
            for (int elem = 0; elem < NelemPerThread; ++elem) {
                out[elem] = type::Div::compute(
                    in[elem], type::Cast::compute<DataType>(nelem));
            }
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename ReduceType, typename _DataType, int _NelemPerThread,
          int Axis>
struct EwiseReduceCompType;

// Conduct reduction on N dimension of the input.
template <typename InDims, typename InShape, typename OutDims,
          typename ReduceType, typename _DataType, int _NelemPerThread>
struct EwiseReduceCompType<InDims, InShape, OutDims, ReduceType, _DataType,
                           _NelemPerThread, AxisType::N> {
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        int idx_out = idx_c * OutDims::HW + idx_h * OutDims::W + idx_w;
        int idx_in = idx_c * InDims::HW + idx_h * InDims::W + idx_w;
        DataType reduced[NelemPerThread];

        ReduceType::template identity<NelemPerThread>(reduced);
#pragma unroll
        for (int i = 0; i < InShape::N; ++i) {
            ReduceType::template reduce<NelemPerThread>(
                reduced, reduced, &in[idx_in + i * InDims::CHW]);
        }
        ReduceType::template postReduce<NelemPerThread>(&out[idx_out], reduced,
                                                        InShape::N);
    }
};

// Conduct reduction on C dimension of the input.
template <typename InDims, typename InShape, typename OutDims,
          typename ReduceType, typename _DataType, int _NelemPerThread>
struct EwiseReduceCompType<InDims, InShape, OutDims, ReduceType, _DataType,
                           _NelemPerThread, AxisType::C> {
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        int idx_out = idx_n * OutDims::CHW + idx_h * OutDims::W + idx_w;
        int idx_in = idx_n * InDims::CHW + idx_h * InDims::W + idx_w;
        DataType reduced[NelemPerThread];

        ReduceType::template identity<NelemPerThread>(reduced);
#pragma unroll
        for (int i = 0; i < InShape::C; ++i) {
            ReduceType::template reduce<NelemPerThread>(
                reduced, reduced, &in[idx_in + i * InDims::HW]);
        }
        ReduceType::template postReduce<NelemPerThread>(&out[idx_out], reduced,
                                                        InShape::C);
    }
};

// Conduct reduction on H dimension of the input.
template <typename InDims, typename InShape, typename OutDims,
          typename ReduceType, typename _DataType, int _NelemPerThread>
struct EwiseReduceCompType<InDims, InShape, OutDims, ReduceType, _DataType,
                           _NelemPerThread, AxisType::H> {
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        int idx_out = idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_w;
        int idx_in = idx_n * InDims::CHW + idx_c * InDims::HW + idx_w;
        DataType reduced[NelemPerThread];

        ReduceType::template identity<NelemPerThread>(reduced);
#pragma unroll
        for (int i = 0; i < InShape::H; ++i) {
            ReduceType::template reduce<NelemPerThread>(
                reduced, reduced, &in[idx_in + i * InDims::W]);
        }
        ReduceType::template postReduce<NelemPerThread>(&out[idx_out], reduced,
                                                        InShape::H);
    }
};

// Conduct reduction on W dimension of the input.
template <typename InDims, typename InShape, typename OutDims,
          typename ReduceType, typename _DataType, int _NelemPerThread>
struct EwiseReduceCompType<InDims, InShape, OutDims, ReduceType, _DataType,
                           _NelemPerThread, AxisType::W> {
    using DataType = _DataType;
    static const int NelemPerThread = _NelemPerThread;

    static DEVICE void compute(DataType *out, DataType *in, int idx_n,
                               int idx_c, int idx_h, int idx_w) {
        int idx_out =
            idx_n * OutDims::CHW + idx_c * OutDims::HW + idx_h * OutDims::W;
        int idx_in =
            idx_n * InDims::CHW + idx_c * InDims::HW + idx_h * InDims::W;
        DataType reduced[NelemPerThread];

        ReduceType::template identity<NelemPerThread>(reduced);
#pragma unroll
        for (int i = 0; i < InShape::W; ++i) {
            ReduceType::template reduce<NelemPerThread>(reduced, reduced,
                                                        &in[idx_in + i]);
        }

        DataType finalSum;
        ReduceType::template identity<1>(&finalSum);
#pragma unroll
        for (int i = 0; i < NelemPerThread; ++i) {
            ReduceType::template reduce<1>(&finalSum, &finalSum, &reduced[i]);
        }
        ReduceType::template postReduce<1>(&out[idx_out], &finalSum,
                                           InShape::W);
    }
};

// Reduce one dimension of input into output.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename ReduceType, int NelemPerThread, int Axis>
struct EwiseReduce {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");

    /// Conduct reduction of the input.
    /// @param out Output tensor.
    /// @param in Input tensor.
    /// @param uop_idx Index of the unit operator.
    template <typename DataType>
    static DEVICE void run(DataType *out, DataType *in, int uop_idx) {
        static_assert(Axis == AxisType::N || Axis == AxisType::C ||
                          Axis == AxisType::H || Axis == AxisType::W,
                      "Invalid reduction axis.");

        using ShapeChecker =
            ReduceShapeChecker<InShape, OutShape, UnitOutDims, Axis>;

        Ewise1<
            OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes,
            EwiseReduceCompType<InDims, InShape, OutDims, ReduceType, DataType,
                                NelemPerThread, Axis>>::run(out, in, uop_idx);
    }
};

// Warp-wise reduction. Only support reduction along the W dimension.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename ReduceType, int NelemPerThread, int Axis>
struct WwiseReduce {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static_assert(UnitOutDims::W % NelemPerThread == 0,
                  "UnitOutDims::W must be divisible by NelemPerThread");
    static_assert(Axis == AxisType::W, "Only support reduction along W axis");

    // TODO(chhwang): support NelemPerThread > 1.
    static_assert(NelemPerThread == 1, "Unimplemented");

    /// Conduct reduction on W dimension of the input.
    /// @param out Output tensor.
    /// @param in Input tensor.
    /// @param uop_idx Index of the unit operator.
    template <typename DataType>
    static DEVICE void runW(DataType *out, DataType *in, int uop_idx,
                            int smem_per_warp) {
        using ShapeChecker =
            ReduceShapeChecker<InShape, OutShape, UnitOutDims, Axis>;

        constexpr int NonReduceDimLength =
            UnitOutDims::N * UnitOutDims::C * UnitOutDims::H;
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

        int idx_out = (tid_h + uh * UnitOutDims::H) * OutDims::W +
                      (tid_c + uc * UnitOutDims::C) * OutDims::HW +
                      (tid_n + un * UnitOutDims::N) * OutDims::CHW;
        int idx_in_base = (tid_h + uh * UnitOutDims::H) * InDims::W +
                          (tid_c + uc * UnitOutDims::C) * InDims::HW +
                          (tid_n + un * UnitOutDims::N) * InDims::CHW;

        DataType reduced[NelemPerThread];

        ReduceType::template identity<NelemPerThread>(reduced);
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_w;
            ReduceType::template reduce<NelemPerThread>(reduced, reduced,
                                                        &in[idx_in]);
        }

        DataType finalSum;
        ReduceType::template identity<1>(&finalSum);
#pragma unroll
        for (int i = 0; i < NelemPerThread; ++i) {
            ReduceType::template reduce<1>(&finalSum, &finalSum, &reduced[i]);
        }

        UnitOp::sync_threads();

        // final reduction on shared memory using warp shuffle.
        finalSum = warpsReduce<ReduceType, UnitOp, ThreadsPerRow>(
            finalSum, tid, smem_per_warp);

        // write the result to output.
        if (tid % ThreadsPerRow == 0) {
            ReduceType::template postReduce<1>(&out[idx_out], &finalSum,
                                               InShape::W);
        }

        UnitOp::sync_threads();
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int Axis, typename DataType>
DEVICE void reduce_e_sum(DataType *out, DataType *in, int uop_idx, int) {
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                SmemBytes, ReduceTypeSum, 1, Axis>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int Axis, typename DataType>
DEVICE void reduce_e_mean(DataType *out, DataType *in, int uop_idx, int) {
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                SmemBytes, ReduceTypeMean, 1, Axis>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int Axis, typename DataType>
DEVICE void reduce_e_max(DataType *out, DataType *in, int uop_idx, int) {
    EwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                SmemBytes, ReduceTypeMax, 1, Axis>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int Axis, typename DataType>
DEVICE void reduce_w_sum(DataType *out, DataType *in, int uop_idx,
                         int smem_per_warp) {
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                SmemBytes, ReduceTypeSum, 1, Axis>::runW(out, in, uop_idx,
                                                         smem_per_warp);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int Axis, typename DataType>
DEVICE void reduce_w_mean(DataType *out, DataType *in, int uop_idx,
                          int smem_per_warp) {
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                SmemBytes, ReduceTypeMean, 1, Axis>::runW(out, in, uop_idx,
                                                          smem_per_warp);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int Axis, typename DataType>
DEVICE void reduce_w_max(DataType *out, DataType *in, int uop_idx,
                         int smem_per_warp) {
    WwiseReduce<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
                SmemBytes, ReduceTypeMax, 1, Axis>::runW(out, in, uop_idx,
                                                         smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_REDUCE_H_
