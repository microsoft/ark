// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EMBEDDING_H_
#define ARK_KERNELS_EMBEDDING_H_

#include "common/broadcast.h"
#include "common/type_intrinsics.h"

namespace ark {

// Rotary Position Embedding(RoPE): https://arxiv.org/pdf/2104.09864.pdf

template <typename DataType>
struct RoPE {
    using InputType = DataType;
    using OutputType = DataType;

    static const int NelemPerThread = 2;

    static DEVICE void compute(DataType *c, const DataType *a,
                               const DataType *b) {
        c[0] = type::Sub::compute(type::Mul::compute(a[0], b[0]),
                                  type::Mul::compute(a[1], b[1]));
        c[1] = type::Add::compute(type::Mul::compute(a[0], b[1]),
                                  type::Mul::compute(a[1], b[0]));
    }
};

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes, typename DataType>
DEVICE void rope(DataType *c, DataType *a, DataType *b, int uop_idx, int) {
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumWarps, SmemBytes, RoPE<DataType>>::run(c, a, b,
                                                                      uop_idx);
}

// struct Rope {
//     static DEVICE float2 compute(float2 a, float2 b) {
//         float2 out;
//         out.x = a.x * b.x - a.y * b.y;
//         out.y = a.x * b.y + a.y * b.x;
//         return out;
//     }
//     static DEVICE __half2 compute(__half2 a, __half2 b) {
//         __half2 out;
//         out.x = __hmul(a.x, b.x) - __hmul(a.y, b.y);
//         out.y = __hmul(a.x, b.y) + __hmul(a.y, b.x);
//         return out;
//     }
//     // struct DEVICE __nv_bfloat162 compute(__nv_bfloat162 a, __nv_bfloat162
//     b)
//     // {
//     //     __nv_bfloat162 out;
//     //     out.x = __hmul(a.x, b.x) - __hmul(a.y, b.y);
//     //     out.y = __hmul(a.x, b.y) + __hmul(a.y, b.x);
//     //     return out;
//     // }
// };

// template <typename In0Dims, typename In0Shape, typename In1Dims,
//           typename In1Shape, typename OutDims, typename OutShape,
//           typename UnitOutDims, int NumWarps, int SmemBytes,
//           typename DataType>
// DEVICE void rope(DataType *c, DataType *a, DataType *b, int uop_idx, int) {
//     static_assert(In0Dims::W % 2 == 0, "");
//     static_assert(In1Dims::W % 2 == 0, "");
//     static_assert(OutDims::W % 2 == 0, "");
//     static_assert(In0Shape::W % 2 == 0, "");
//     static_assert(In1Shape::W % 2 == 0, "");
//     static_assert(OutShape::W % 2 == 0, "");

//     using VecType = typename std::conditional<
//         std::is_same<DataType, float>::value, float2,
//         typename std::conditional<
//             std::is_same<DataType, half>::value, __half2,
//             typename std::conditional<std::is_same<DataType,
//             bfloat16>::value,
//                                       __nv_bfloat162,
//                                       void>::type>::type>::type;

//     using In0VecDims = Vec<In0Dims::N, In0Dims::C, In0Dims::H, In0Dims::W /
//     2>; using In1VecDims = Vec<In1Dims::N, In1Dims::C, In1Dims::H, In1Dims::W
//     / 2>; using OutVecDims = Vec<OutDims::N, OutDims::C, OutDims::H,
//     OutDims::W / 2>;

//     using In0VecShape = Vec<In0Shape::N, In0Shape::C, In0Shape::H,
//                             In0Shape::W / 2>;
//     using In1VecShape = Vec<In1Shape::N, In1Shape::C, In1Shape::H,
//                             In1Shape::W / 2>;
//     using OutVecShape = Vec<OutShape::N, OutShape::C, OutShape::H,
//                             OutShape::W / 2>;

//     Broadcast2<In0VecDims, In0VecShape, In1VecDims, In1VecShape, OutVecDims,
//     OutVecShape,
//                UnitOutDims, NumWarps, SmemBytes,
//                Broadcast2Intrinsic<Rope, In0VecShape, In1VecShape,
//                VecType,
//                                    VecType, 1>>::run((VecType *)c, (VecType
//                                    *)a,
//                                                      (VecType *)b, uop_idx);
// }

// Embedding

template <typename _DataType>
struct Assign {
    using InputType = _DataType;
    using OutputType = _DataType;
    static const int NelemPerThread = 1;
    static DEVICE void compute(_DataType *y, const _DataType *x) { *y = *x; }
};

template <typename InDims, typename InShape, typename WeightDims,
          typename WeightShape, typename OutDims, typename OutShape,
          int EmbeddingDim, int NumWarps, typename DataType>
DEVICE void embedding(DataType *output, int *input, DataType *weight,
                      int uop_idx, int) {
    // InShape:     Vec<D0, D1, D2, 1>
    // WeightShape: Vec< 1,  1,  ?, EmbeddingDim> (?: # of embeddings)
    // OutShape:    Vec<D0, D1, D2, EmbeddingDim>

    static_assert(InShape::W == 1, "");

    using UnitOutDims = Vec<1, 1, 1, OutDims::W>;
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, 0>;
    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);
    int uh = UnitOp::uop_idx_h(uop_idx);

    // pWeight: Vec<1, 1, 1, EmbeddingDim>
    int emb_idx = input[un * InDims::CH + uc * InDims::H + uh];
    if (emb_idx < 0) {
        emb_idx += WeightShape::H;
    }
    // TODO: assert if emb_idx is still negative
    DataType *pWeight = &weight[emb_idx * WeightDims::W];

    Broadcast1<Vec<1, 1, 1, WeightDims::W>, Vec<1, 1, 1, EmbeddingDim>, OutDims,
               OutShape, UnitOutDims, NumWarps, 0,
               Assign<DataType>>::run(output, pWeight, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_EMBEDDING_H_
