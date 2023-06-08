// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_LAYERNORM_H_
#define ARK_KERNELS_LAYERNORM_H_

#include "reduce.h"

namespace ark {


// Perform layer normalization on input and write the result on output.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, typename ReduceType, typename DataType,
          int NelemPerThread>
struct Layernorm
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;
    
    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static Device void run(float *out, float *in, int tx, int ty, int tz)
    {
        int tid = UnitOp::thread_id();
        __shared__ float s_mean[ThreadsNum];
    __shared__ float s_var[ThreadsNum];

    }
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes>
DEVICE void layernorm(float *out, float *in, int tx, int ty, int tz)
{
    Layernorm<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
              SmemBytes, ReduceType, DataType, NelemPerThread>::run(
        out, in, tx, ty, tz);
}



} // namespace ark

#endif // ARK_KERNELS_LAYERNORM_H_
