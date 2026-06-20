// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_CACHE_H_
#define ARK_KERNELS_CACHE_H_

#include <stdint.h>

#include "common/unit_op.h"

namespace ark {

DEVICE void kv_cache_slot_bounds_trap() {
#if defined(ARK_TARGET_CUDA_ARCH)
    asm volatile("trap;");
#elif defined(ARK_TARGET_ROCM_ARCH)
    __builtin_trap();
#endif
}

template <int MaxSeq, typename SlotDims, typename SlotShape,
          typename UnitSlotDims, int NumWarps, int SmemBytes, typename DataType>
DEVICE void kv_cache_slot(DataType *out, DataType *cache, const DataType *token,
                          int32_t *position, int uop_idx,
                          [[maybe_unused]] int smem_per_warp) {
    using SlotUnitOp = ark::UnitOp<SlotDims, SlotShape, UnitSlotDims,
                                   NumWarps, SmemBytes>;

    // The model config fixes NumTasks=1. Keep a guard here so a future manual
    // config cannot advance the shared position once per task.
    if (uop_idx != 0) return;

    const int pos = position[0];
    SlotUnitOp::sync_threads();
    if (pos < 0 || pos >= MaxSeq) {
        if (SlotUnitOp::thread_id() == 0) {
            position[0] = INT32_MIN;
            kv_cache_slot_bounds_trap();
        }
        return;
    }

    static constexpr size_t StepSize = SlotUnitOp::NumThreads;
    const size_t slot_base = static_cast<size_t>(pos) * SlotShape::NCHW;

    for (size_t idx = SlotUnitOp::thread_id(); idx < SlotShape::NCHW;
         idx += StepSize) {
        cache[slot_base + idx] = token[idx];
    }

    SlotUnitOp::sync_threads();

    for (size_t idx = SlotUnitOp::thread_id(); idx < SlotShape::NCHW;
         idx += StepSize) {
        out[idx] = cache[slot_base + idx];
    }

    SlotUnitOp::sync_threads();

    if (SlotUnitOp::thread_id() == 0) {
        position[0] = pos + 1;
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_CACHE_H_
