// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EMBEDDING_H_
#define ARK_KERNELS_EMBEDDING_H_

#include "common.h"

namespace ark {

// Rotary Position Embedding(RoPE): https://arxiv.org/pdf/2104.09864.pdf

template <typename DataType> struct RoPE;

template <> struct RoPE<float>
{
    using InputType = float;
    using OutputType = float;
    static const int NelemPerThread = 2;
    static DEVICE void compute(float *c, const float *a, const float *b)
    {
        float2 *pc = (float2 *)c;
        const float2 *pa = (const float2 *)a;
        const float2 *pb = (const float2 *)b;
        pc->x = pa->x * pb->x - pa->y * pb->y;
        pc->y = pa->x * pb->y + pa->y * pb->x;
    }
};

template <> struct RoPE<half>
{
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 2;
    static DEVICE void compute(half *c, const half *a, const half *b)
    {
        __half2 *pc = (__half2 *)c;
        const __half2 *pa = (const __half2 *)a;
        const __half2 *pb = (const __half2 *)b;
        pc->x = __hmul(pa->x, pb->x) - __hmul(pa->y, pb->y);
        pc->y = __hmul(pa->x, pb->y) + __hmul(pa->y, pb->x);
    }
};

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void rope(float *c, float *a, float *b, int uop_idx, int)
{
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes, RoPE<float>>::run(c, a, b,
                                                                     uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void rope(half *c, half *a, half *b, int uop_idx, int)
{
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes, RoPE<half>>::run(c, a, b,
                                                                    uop_idx);
}

// Embedding

template <typename _DataType> struct Assign
{
    using InputType = _DataType;
    using OutputType = _DataType;
    static const int NelemPerThread = 1;
    static DEVICE void compute(_DataType *c, const _DataType *a)
    {
        *c = *a;
    }
};

template <typename InDims, typename InShape, typename WeightDims,
          typename WeightShape, typename OutDims, typename OutShape,
          int EmbeddingDim, int NumThreads>
DEVICE void embedding(float *output, int *input, float *weight, int uop_idx,
                      int)
{
    // InShape:     Vec<D0, D1, D2, 1>
    // WeightShape: Vec< 1,  1,  ?, EmbeddingDim> (?: # of embeddings)
    // OutShape:    Vec<D0, D1, D2, EmbeddingDim>

    static_assert(InShape::W == 1, "");

    using UnitOutDims = Vec<1, 1, 1, OutDims::W>;
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, 0>;
    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);
    int uh = UnitOp::uop_idx_h(uop_idx);

    // pWeight: Vec<1, 1, 1, EmbeddingDim>
    int emb_idx = input[un * InDims::CH + uc * InDims::H + uh];
    float *pWeight = &weight[emb_idx * WeightDims::W];

    Broadcast1<Vec<1, 1, 1, WeightDims::W>, Vec<1, 1, 1, EmbeddingDim>, OutDims,
               OutShape, UnitOutDims, NumThreads, 0,
               Assign<float>>::run(output, pWeight, uop_idx);
}

template <typename InDims, typename InShape, typename WeightDims,
          typename WeightShape, typename OutDims, typename OutShape,
          int EmbeddingDim, int NumThreads>
DEVICE void embedding(half *output, int *input, half *weight, int uop_idx, int)
{
    // InShape:     Vec<D0, D1, D2, 1>
    // WeightShape: Vec< 1,  1,  ?, EmbeddingDim> (?: # of embeddings)
    // OutShape:    Vec<D0, D1, D2, EmbeddingDim>

    static_assert(InShape::W == 1, "");

    using UnitOutDims = Vec<1, 1, 1, OutDims::W>;
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, 0>;
    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);
    int uh = UnitOp::uop_idx_h(uop_idx);

    // pWeight: Vec<1, 1, 1, EmbeddingDim>
    int emb_idx = input[un * InDims::CH + uc * InDims::H + uh];
    half *pWeight = &weight[emb_idx * WeightDims::W];

    Broadcast1<Vec<1, 1, 1, WeightDims::W>, Vec<1, 1, 1, EmbeddingDim>, OutDims,
               OutShape, UnitOutDims, NumThreads, 0,
               Assign<half>>::run(output, pWeight, uop_idx);
}

} // namespace ark

#endif // ARK_KERNELS_EMBEDDING_H_
