// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifdef __CUDACC__
#ifndef ARK_KERNELS_H_
#define ARK_KERNELS_H_

extern __shared__ int _ARK_SMEM[];

// clang-format off
#include "half.h"
// clang-format on

#include "activation.h"
#include "arithmetic.h"
#include "comm.h"
#include "comm_mm.h"
#include "im2col.h"
#include "layernorm.h"
#include "matmul.h"
#include "reduce.h"
#include "smem.h"
#include "softmax.h"
#include "sync.h"
#include "transpose.h"

#endif // ARK_KERNELS_H_
#endif // __CUDACC__
