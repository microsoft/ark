// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_KERNELS_SYNC_H_
#define ARK_KERNELS_SYNC_H_

#include "static_math.h"

#define ARK_KERNELS_SYNC_GPU_TEST 0

namespace ark {

namespace sync {

struct State
{
    volatile int flag;
    int cnt;
    int is_add;
    int clks_cnt;
};

} // namespace sync

#if (ARK_KERNELS_SYNC_GPU_TEST)
__device__ volatile int TMP[80];

template <int BlockNum> DEVICE void sync_gpu(sync::State &state)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        int is_add_ = state.is_add ^ 1;
        int idx = blockIdx.x >> 1;
        volatile long int *pTMP = (volatile long int *)TMP;
        if (blockIdx.x & 1 || blockIdx.x >= 40) {
            // 1, 3, 5, 7, ..., 79
            TMP[blockIdx.x] = is_add_ + 1;
            goto Barrier;
        }
        if (is_add_ == 0) {
            while (pTMP[idx + 20] != 4294967297) {
            }
            if (idx >= 10) {
                TMP[blockIdx.x] = 1;
                goto Barrier;
            }
            while (pTMP[idx + 10] != 4294967297) {
            }
            if (idx >= 5) {
                TMP[blockIdx.x] = 1;
                goto Barrier;
            }
            while (pTMP[idx + 5] != 4294967297) {
            }
            if (idx >= 2) {
                TMP[blockIdx.x] = 1;
                goto Barrier;
            }
            while (pTMP[idx + 2] != 4294967297) {
            }
            if (idx >= 1) {
                TMP[blockIdx.x] = 1;
                goto Barrier;
            }
            while (pTMP[1] != 4294967297) {
            }
            while (pTMP[4] != 4294967297) {
            }
        } else {
            while (pTMP[idx + 20] != 8589934594) {
            }
            if (idx >= 10) {
                TMP[blockIdx.x] = 2;
                goto Barrier;
            }
            while (pTMP[idx + 10] != 8589934594) {
            }
            if (idx >= 5) {
                TMP[blockIdx.x] = 2;
                goto Barrier;
            }
            while (pTMP[idx + 5] != 8589934594) {
            }
            if (idx >= 2) {
                TMP[blockIdx.x] = 2;
                goto Barrier;
            }
            while (pTMP[idx + 2] != 8589934594) {
            }
            if (idx >= 1) {
                TMP[blockIdx.x] = 2;
                goto Barrier;
            }
            while (pTMP[1] != 8589934594) {
            }
            while (pTMP[4] != 8589934594) {
            }
        }
    Barrier:
        //
        if (blockIdx.x == 0) {
            state.flag ^= 1;
        } else {
            while (state.flag != is_add_) {
            }
        }
        state.is_add = is_add_;
    }
    __syncthreads();
}
#else // (ARK_KERNELS_SYNC_GPU_TEST)
// Synchronize multiple thread blocks inside a kernel. Guarantee that all
// previous work of all threads in cooperating blocks is finished and
// visible to all threads in the device.
template <int BlockNum> DEVICE void sync_gpu(sync::State &state)
{
    constexpr int MaxOldCnt = BlockNum - 1;
    __threadfence();
    // Make sure that all threads in this block have done `__threadfence()`
    // before to flip `flag`.
#ifdef ARK_KERNELS_SYNC_CLKS_CNT
    static_assert(math::is_pow2<ARK_KERNELS_SYNC_CLKS_CNT>::value == 1, "");
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        _ARK_CLKS[state.clks_cnt] = clock64();
        state.clks_cnt = (state.clks_cnt + 1) & (ARK_KERNELS_SYNC_CLKS_CNT - 1);
    }
#endif // ARK_KERNELS_SYNC_CLKS_CNT
    __syncthreads();
    if (threadIdx.x == 0) {
        int is_add_ = state.is_add ^ 1;
        if (is_add_) {
            if (atomicAdd(&state.cnt, 1) == MaxOldCnt) {
                state.flag = 1;
            }
            while (!state.flag) {
            }
        } else {
            if (atomicSub(&state.cnt, 1) == 1) {
                state.flag = 0;
            }
            while (state.flag) {
            }
        }
        state.is_add = is_add_;
    }
    // We need sync here because only a single thread is checking whether
    // the flag is flipped.
    __syncthreads();
}
#endif // (ARK_KERNELS_SYNC_GPU_TEST)

// Synchronize a group of warps.
// This function replaces `__syncthreads()` of legacy kernel implementations.
// It is needed because in normal practices to implement a kernel, each thread
// block typically processes a single unit task (e.g. one tile) so it is common
// to synchronize all co-working threads via `__syncthreads()`, however in our
// case, we often run multiple tasks or tiles in a single thread block, so we
// need a function which lets each tile to synchronize their own using threads
// only. Since `__syncthreads()` synchronize the entire threads in a thread
// block, we implement a finer-grained version of this via `barrier.sync` PTX
// instruction.
template <int ThreadsPerWarpGroup> DEVICE void sync_warps()
{
    static_assert(ThreadsPerWarpGroup == 32 || ThreadsPerWarpGroup == 64 ||
                      ThreadsPerWarpGroup == 128 ||
                      ThreadsPerWarpGroup == 256 ||
                      ThreadsPerWarpGroup == 512 || ThreadsPerWarpGroup == 1024,
                  "");
    // When ThreadsPerWarpGroup is 64, this function should not be called in
    // parallel with __syncthreads(). The following is the explanation why.
    // GPUs have 16 hardware barriers, numbered 0~15. This means that more than
    // sixteen `barrier.sync` instructions cannot run at the same time due to HW
    // limitation. This is reasonable because the maximum threads per block is
    // 1024 (32 warps), so we need at most 16 barriers, which happens in the
    // case when we synchronize warps in two-pairs. If we synchronize warps in
    // four-pairs, we need at most 8 barriers. The problem here is that
    // `__syncthreads()` always uses barrier 0, so if `__syncthreads()`
    // instruction is on flight, `sync_wg()` should not use barrier 0. However,
    // we cannot know whether `__syncthreads()` will be on flight or not.
    // So we have two options.
    // Option 1: Let users take the risk and we just use barrier 0, which
    // enables to support 64 threads (sync in two-pairs). In this case, users
    // should make sure that their kernels never issue `__syncthreads()` and
    // `sync_wg()` at the same time, otherwise the kernel may stop unexpectedly
    // during runtime.
    // Option 2: Do not use barrier 0 to be more safe, instead we cannot support
    // 64 threads.
    // Here we selected the first option
    if (ThreadsPerWarpGroup == 32) {
        __syncwarp();
    } else if (ThreadsPerWarpGroup == 64) {
        asm volatile("barrier.sync %0, 64;" ::"r"((threadIdx.x >> 6)));
    } else if (ThreadsPerWarpGroup == 128) {
        asm volatile("barrier.sync %0, 128;" ::"r"((threadIdx.x >> 7) + 8));
    } else if (ThreadsPerWarpGroup == 256) {
        asm volatile("barrier.sync %0, 256;" ::"r"((threadIdx.x >> 8) + 8));
    } else if (ThreadsPerWarpGroup == 512) {
        asm volatile("barrier.sync %0, 512;" ::"r"((threadIdx.x >> 9) + 8));
    } else if (ThreadsPerWarpGroup == 1024) {
        // If we sync 1024 threads, it means we sync all threads in a thread
        // block because the maximum number of threads per block is 1024 in
        // all NVIDIA devices. Therefore, we do not check `threadIdx.x` and
        // just use barrier 8.
        asm volatile("barrier.sync 8, 1024;");
    }
}

} // namespace ark

#endif // ARK_KERNELS_SYNC_H_
