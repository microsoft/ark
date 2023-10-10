// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mgr.h"

#include "include/ark.h"
#include "include/ark_utils.h"
#include "unittest/unittest_utils.h"

using namespace std;
using namespace ark;

// Test initializing and destroying GpuMgr and GpuMgrCtx.
unittest::State test_gpu_mgr_basic() {
    int pid = ark::utils::proc_spawn([] {
        unittest::Timeout timeout{10};
        GpuMgr *mgr_a = get_gpu_mgr(0);
        GpuMgr *mgr_b = get_gpu_mgr(0);

        UNITTEST_EQ(mgr_a, mgr_b);

        GpuMgr *mgr_c = get_gpu_mgr(1);

        UNITTEST_NE(mgr_a, mgr_c);

        GpuMgrCtx *ctx_a = mgr_a->create_context("test_a", 0, 1);
        UNITTEST_NE(ctx_a, (GpuMgrCtx *)nullptr);
        UNITTEST_EQ(ctx_a->get_name(), "test_a");
        UNITTEST_EQ(ctx_a->get_rank(), 0);
        UNITTEST_EQ(ctx_a->get_world_size(), 1);
        UNITTEST_EQ(ctx_a->get_gpu_id(), 0);

        mgr_a->destroy_context(ctx_a);

        return 0;
    });

    int ret = ark::utils::proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

unittest::State test_gpu_mgr_mem_alloc() {
    int pid = ark::utils::proc_spawn([] {
        unittest::Timeout timeout{10};
        GpuMgr *mgr = get_gpu_mgr(0);

        GpuMgrCtx *ctx = mgr->create_context("test", 0, 1);
        UNITTEST_NE(ctx, (GpuMgrCtx *)nullptr);

        GpuBuf *buf0 = ctx->mem_alloc(sizeof(int));
        GpuBuf *buf1 = ctx->mem_alloc(sizeof(int));

        UNITTEST_TRUE(ctx->get_total_bytes() >= 2 * sizeof(int));

        ctx->freeze();

        int buf0_data = 7;
        int buf1_data = 8;
        gpu_memcpy(buf0, &buf0_data, sizeof(int));
        gpu_memcpy(buf1, &buf1_data, sizeof(int));

        int buf0_data2 = 0;
        int buf1_data2 = 0;

        gpu_memcpy(&buf0_data2, buf0, sizeof(int));
        gpu_memcpy(&buf1_data2, buf1, sizeof(int));

        UNITTEST_EQ(buf0_data2, 7);
        UNITTEST_EQ(buf1_data2, 8);

        gpu_memcpy(buf0, buf1, sizeof(int));

        gpu_memcpy(&buf0_data2, buf0->ref(), sizeof(int));
        UNITTEST_EQ(buf0_data2, 8);

        mgr->destroy_context(ctx);

        return 0;
    });

    int ret = ark::utils::proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

unittest::State test_gpu_mgr_mem_free() {
    int pid = ark::utils::proc_spawn([] {
        unittest::Timeout timeout{10};
        GpuMgr *mgr = get_gpu_mgr(0);

        GpuMgrCtx *ctx = mgr->create_context("test", 0, 1);
        UNITTEST_NE(ctx, (GpuMgrCtx *)nullptr);

        GpuBuf *buf0 = ctx->mem_alloc(sizeof(int));

        // This does not mean to free the memory, but means that the following
        // `mem_alloc()` can reuse the memory of `buf0`.
        ctx->mem_free(buf0);

        // This should reuse the memory of `buf0`.
        GpuBuf *buf1 = ctx->mem_alloc(sizeof(int));

        ctx->freeze();

        UNITTEST_EQ(buf0->ref(), buf1->ref());
        UNITTEST_EQ(buf0->href(), buf1->href());
        UNITTEST_EQ(buf0->pref(), buf1->pref());

        int buf0_data = 9;
        gpu_memcpy(buf0, &buf0_data, sizeof(int));

        int buf1_data = 0;

        gpu_memcpy(&buf1_data, buf1, sizeof(int));

        UNITTEST_EQ(buf1_data, 9);

        mgr->destroy_context(ctx);

        return 0;
    });

    int ret = ark::utils::proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

// Test accessing remote GPU's memory space.
unittest::State test_gpu_mgr_remote() {
    int pid0 = ark::utils::proc_spawn([] {
        unittest::Timeout timeout{10};
        GpuMgr *mgr = get_gpu_mgr(0);
        GpuMgrCtx *ctx = mgr->create_context("test", 0, 2);

        GpuBuf *gpu0_eid3 = ctx->mem_alloc(sizeof(int));
        GpuBuf *gpu0_eid4 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu0_eid3, 0, 3);
        ctx->mem_export(gpu0_eid4, 0, 4);

        GpuBuf *gpu1_eid5 = ctx->mem_import(sizeof(int), 5, 1);
        GpuBuf *gpu1_eid6 = ctx->mem_import(sizeof(int), 6, 1);

        ctx->freeze(true);

        volatile int *ptr = (volatile int *)gpu0_eid3->href();
        while (*ptr != 7890) {
        }

        gpu_memset(gpu1_eid5, 1234, 1);

        ptr = (volatile int *)gpu0_eid4->href();
        while (*ptr != 3456) {
        }

        gpu_memset(gpu1_eid6, 5678, 1);

        mgr->destroy_context(ctx);
        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::utils::proc_spawn([] {
        unittest::Timeout timeout{10};
        GpuMgr *mgr = get_gpu_mgr(1);
        GpuMgrCtx *ctx = mgr->create_context("test", 1, 2);

        GpuBuf *gpu1_eid5 = ctx->mem_alloc(sizeof(int));
        GpuBuf *gpu1_eid6 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu1_eid5, 0, 5);
        ctx->mem_export(gpu1_eid6, 0, 6);

        GpuBuf *gpu0_eid3 = ctx->mem_import(sizeof(int), 3, 0);
        GpuBuf *gpu0_eid4 = ctx->mem_import(sizeof(int), 4, 0);

        ctx->freeze(true);

        gpu_memset(gpu0_eid3, 7890, 1);

        volatile int *ptr = (volatile int *)gpu1_eid5->href();
        while (*ptr != 1234) {
        }

        gpu_memset(gpu0_eid4->ref(), 3456, 1);

        ptr = (volatile int *)gpu1_eid6->href();
        while (*ptr != 5678) {
        }

        mgr->destroy_context(ctx);
        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = ark::utils::proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gpu_mgr_basic);
    UNITTEST(test_gpu_mgr_mem_alloc);
    UNITTEST(test_gpu_mgr_mem_free);
    UNITTEST(test_gpu_mgr_remote);
    return 0;
}
