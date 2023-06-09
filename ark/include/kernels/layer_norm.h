// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_LAYER_NORM_H_
#define ARK_KERNELS_LAYER_NORM_H_

// #include "LayerNorm.h"
#include "base_op.h"
#include "static_math.h"
#include "transform.h"

namespace ark {

// Static checkers if InShape can be reduced into OutShape.
template <typename InShape, typename OutShape> struct LayerNormShapeChecker
{
    static_assert(InShape::N == OutShape::N,
                  "Dimension N of input and output do not match");
    static_assert(InShape::C == OutShape::C,
                  "Dimension C of input and output do not match");
    static_assert(InShape::H == OutShape::H,
                  "Dimension H of input and output do not match");
    static_assert(OutShape::W == OutShape::W,
                  "Dimension W of input and output do not match");
};

// Perform layer normalization on input and write the result on output.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, typename DataType, int NelemPerThread>
struct LayerNorm
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static DEVICE void run(DataType *out, DataType *in, int tx, int ty, int tz)
    {
        using InOutChk = LayerNormShapeChecker<InShape, OutShape>;

        constexpr int NonLayerNormDimLength =
            UnitOutShape::N * UnitOutShape::C * UnitOutShape::H;
        // The reduction dimension of the final stage.
        // Assume this division is always exact.
        static_assert((ThreadsNum * NelemPerThread) % NonLayerNormDimLength ==
                      0);
        constexpr int FinalDim =
            (ThreadsNum * NelemPerThread) / NonLayerNormDimLength;

        //         // Shared memory
        //         LayerNormSharedStorage<DataType, ThreadsNum> *smem =
        //             UnitOp::template shared_memory<
        //                 LayerNormSharedStorage<DataType, ThreadsNum>>();

        int tid = UnitOp::thread_id();
        int tid_w = (tid * NelemPerThread) % FinalDim;
        int tid_h = ((tid * NelemPerThread) / FinalDim) % UnitOutShape::H;
        int tid_c = ((tid * NelemPerThread) / FinalDim / UnitOutShape::H) %
                    UnitOutShape::C;
        int tid_n = (tid * NelemPerThread) / FinalDim / UnitOutShape::H /
                    UnitOutShape::C;

        int idx_out = (tid_h + th * UnitOutShape::H) * OutDims::W +
                      (tid_c + tc * UnitOutShape::C) * OutDims::W * OutDims::H +
                      (tid_n + tn * UnitOutShape::N) * OutDims::W * OutDims::H *
                          OutDims::C;
        int idx_in_base =
            (tid_h + th * UnitOutShape::H) * InDims::W +
            (tid_c + tc * UnitOutShape::C) * InDims::W * InDims::H +
            (tid_n + tn * UnitOutShape::N) * InDims::W * InDims::H * InDims::C;

        //         // smem->storage[tid] = LayerNormType::template
        //         identity<DataType>();
        // calculate the sum of the input along the last dimension W.
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += FinalDim) {
            int idx_in = idx_in_base + idx_in_w;
            DataType sum = in[idx_in];
#pragma unroll
            for (int i = 1; i < NelemPerThread; i++) {
                sum = sum + in[idx_in + i];
            }
            // smem->storage[tid] =
            //     LayerNormType::LayerNorm(smem->storage[tid],
            //     LayerNormd);
        }
        __syncthreads();
        mean = sum / UnitOutShape::W;
        //         // final reduction on shared memory using warp shuffle.
        // DataType val = smem->storage[tid];
        //         val = shfl<LayerNormType, DataType, FinalDim>(val);
        //         val = LayerNormType::postLayerNorm(val, NelemPerThread);
        //         if (tid % FinalDim == 0) {
        //             out[idx_out] = val;
        //         }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes>
DEVICE void layer_norm(float *out, float *in, int tx, int ty, int tz)
{
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
              SmemBytes, float, NelemPerThread>::run(out, in, tx, ty, tz);
}

} // namespace ark

#endif // ARK_KERNELS_LAYER_NORM_H_
