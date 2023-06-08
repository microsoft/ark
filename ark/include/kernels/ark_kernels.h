// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifdef __CUDACC__
#ifndef ARK_KERNELS_H_
#define ARK_KERNELS_H_

// clang-format off
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
// clang-format on

namespace ark {
using half = cutlass::half_t;
} // namespace ark

extern __shared__ int _ARK_SMEM[];

#include "activation.h"
#include "arithmetic.h"
#include "comm.h"
#include "comm_mm.h"
#include "im2col.h"
#include "layer_norm.h"
#include "matmul.h"
#include "reduce.h"
#include "smem.h"
#include "sync.h"
#include "transpose.h"

#endif // ARK_KERNELS_H_
#endif // __CUDACC__
