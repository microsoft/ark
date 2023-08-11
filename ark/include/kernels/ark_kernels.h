// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifdef __CUDACC__
#ifndef ARK_KERNELS_H_
#define ARK_KERNELS_H_

// clang-format off
#include "common.h"
// clang-format on

#include "activation.h"
#include "arithmetic.h"
#include "comm.h"
#include "comm_mm.h"
#include "comm_mscclpp.h"
#include "im2col.h"
#include "layernorm.h"
#include "math_functions.h"
#include "matmul.h"
#include "reduce.h"
#include "softmax.h"
#include "transpose.h"

#endif // ARK_KERNELS_H_
#endif // __CUDACC__
