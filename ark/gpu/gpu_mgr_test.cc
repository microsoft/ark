// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_mgr.h"
#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;
using namespace ark;

// Test initializing and destroying GpuMgr and GpuMgrCtx.
unittest::State test_gpu_mgr_basic()
{
    int pid = proc_spawn([] {
        unittest::Timeout timeout{3};
        GpuMgr *mgr_a = get_gpu_mgr(0);
        GpuMgr *mgr_b = get_gpu_mgr(0);

        UNITTEST_EQ(mgr_a, mgr_b);

        GpuMgr *mgr_c = get_gpu_mgr(1);

        UNITTEST_NE(mgr_a, mgr_c);

        GpuMgrCtx *ctx_a = mgr_a->create_context("test_a", 0, 1);
        UNITTEST_NE(ctx_a, (GpuMgrCtx *)nullptr);

        mgr_a->destroy_context(ctx_a);

        return 0;
    });

    int ret = proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

// Test accessing remote GPU's memory space.
unittest::State test_gpu_mgr_remote()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{5};
        GpuMgr *mgr = get_gpu_mgr(0);
        GpuMgrCtx *ctx = mgr->create_context("test", 0, 2);

        GpuBuf *gpu0_eid3 = ctx->mem_alloc(sizeof(int));
        GpuBuf *gpu0_eid4 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu0_eid3, 0, 3);
        ctx->mem_export(gpu0_eid4, 0, 4);

        GpuBuf *gpu1_eid5 = ctx->mem_import(sizeof(int), 5, 1);
        GpuBuf *gpu1_eid6 = ctx->mem_import(sizeof(int), 6, 1);

        ctx->freeze();

        volatile int *ptr = (volatile int *)gpu0_eid3->href();
        while (*ptr != 7890) {
        }

        gpu_memset(gpu1_eid5, 1234, 1);

        ptr = (volatile int *)gpu0_eid4->href();
        while (*ptr != 3456) {
        }

        gpu_memset(gpu1_eid6, 5678, 1);
        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{5};
        GpuMgr *mgr = get_gpu_mgr(1);
        GpuMgrCtx *ctx = mgr->create_context("test", 1, 2);

        GpuBuf *gpu1_eid5 = ctx->mem_alloc(sizeof(int));
        GpuBuf *gpu1_eid6 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu1_eid5, 0, 5);
        ctx->mem_export(gpu1_eid6, 0, 6);

        GpuBuf *gpu0_eid3 = ctx->mem_import(sizeof(int), 3, 0);
        GpuBuf *gpu0_eid4 = ctx->mem_import(sizeof(int), 4, 0);

        ctx->freeze();

        gpu_memset(gpu0_eid3, 7890, 1);

        volatile int *ptr = (volatile int *)gpu1_eid5->href();
        while (*ptr != 1234) {
        }

        gpu_memset(gpu0_eid4, 3456, 1);

        ptr = (volatile int *)gpu1_eid6->href();
        while (*ptr != 5678) {
        }

        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

// Test accessing remote GPU's memory space after the context is freezed.
unittest::State test_gpu_mgr_remote_lazy_import()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{5};
        GpuMgr *mgr = get_gpu_mgr(0);
        GpuMgrCtx *ctx = mgr->create_context("test", 0, 2);

        GpuBuf *gpu0_eid3 = ctx->mem_alloc(sizeof(int));
        GpuBuf *gpu0_eid4 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu0_eid3, 0, 3);
        ctx->mem_export(gpu0_eid4, 0, 4);

        ctx->freeze();

        GpuBuf *gpu1_eid5 = ctx->mem_import(sizeof(int), 5, 1);
        GpuBuf *gpu1_eid6 = ctx->mem_import(sizeof(int), 6, 1);

        volatile int *ptr = (volatile int *)gpu0_eid3->href();
        while (*ptr != 7890) {
        }

        gpu_memset(gpu1_eid5, 1234, 1);

        ptr = (volatile int *)gpu0_eid4->href();
        while (*ptr != 3456) {
        }

        gpu_memset(gpu1_eid6, 5678, 1);

        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{5};
        GpuMgr *mgr = get_gpu_mgr(1);
        GpuMgrCtx *ctx = mgr->create_context("test", 1, 2);

        GpuBuf *gpu1_eid5 = ctx->mem_alloc(sizeof(int));
        GpuBuf *gpu1_eid6 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu1_eid5, 0, 5);
        ctx->mem_export(gpu1_eid6, 0, 6);

        ctx->freeze();

        GpuBuf *gpu0_eid3 = ctx->mem_import(sizeof(int), 3, 0);
        GpuBuf *gpu0_eid4 = ctx->mem_import(sizeof(int), 4, 0);

        gpu_memset(gpu0_eid3, 7890, 1);

        volatile int *ptr = (volatile int *)gpu1_eid5->href();
        while (*ptr != 1234) {
        }

        gpu_memset(gpu0_eid4, 3456, 1);

        ptr = (volatile int *)gpu1_eid6->href();
        while (*ptr != 5678) {
        }

        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

// Test inter-GPU communication via sending a doorbell.
unittest::State test_gpu_mgr_doorbell()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{5};
        GpuMgr *mgr = get_gpu_mgr(0);
        GpuMgrCtx *ctx = mgr->create_context("test", 0, 2);

        GpuBuf *gpu0_src = ctx->mem_alloc(1024 * sizeof(int));
        ctx->mem_export(gpu0_src, 0, 7);
        ctx->mem_import(1024 * sizeof(int), 5, 1);

        ctx->freeze();

        // Set source data.
        int *data = (int *)gpu0_src->href();
        for (int i = 0; i < 1024; ++i) {
            data[i] = i + 1;
        }

        // Send a doorbell.
        ctx->send(7, 5, 1, 1024 * sizeof(int));

        // Wait for the send completion.
        volatile int *sc = ctx->get_sc_href(7);
        while (*sc == 0) {
        }

        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{5};
        GpuMgr *mgr = get_gpu_mgr(1);
        GpuMgrCtx *ctx = mgr->create_context("test", 1, 2);

        GpuBuf *gpu1_dst = ctx->mem_alloc(1024 * sizeof(int));
        ctx->mem_export(gpu1_dst, 0, 5);
        ctx->mem_import(1024 * sizeof(int), 7, 0);

        ctx->freeze();

        // Wait for the receive completion.
        volatile int *rc = ctx->get_rc_href(5);
        while (*rc == 0) {
        }

        // Verify the received data.
        int *data = (int *)gpu1_dst->href();
        for (int i = 0; i < 1024; ++i) {
            UNITTEST_EQ(data[i], i + 1);
        }

        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_gpu_mgr_basic);
    UNITTEST(test_gpu_mgr_remote);
    UNITTEST(test_gpu_mgr_remote_lazy_import);
    UNITTEST(test_gpu_mgr_doorbell);
    return 0;
}
