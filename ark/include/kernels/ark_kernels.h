// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#if defined(ARK_TARGET_CUDA_ARCH) || defined(ARK_TARGET_ROCM_ARCH)

#ifndef ARK_KERNELS_H_
#define ARK_KERNELS_H_

#include "arithmetic.h"
#include "cast.h"
#include "comm.h"
#include "copy.h"
#include "embedding.h"
#include "im2col.h"
#include "layernorm.h"
#include "math_functions.h"
#include "matmul.h"
#include "noop.h"
#include "reduce.h"
#include "transpose.h"

#endif  // ARK_KERNELS_H_

#endif
