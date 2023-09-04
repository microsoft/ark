// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mgr.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "unittest/unittest_utils.h"

using namespace std;
using namespace ark;

// Test initializing and destroying GpuMgr and GpuMgrCtx.
unittest::State test_gpu_mgr_basic()
{
    int pid = ark::utils::proc_spawn([] {
        unittest::Timeout timeout{10};
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

    int ret = ark::utils::proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

// Test accessing remote GPU's memory space.
unittest::State test_gpu_mgr_remote()
{
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

        ctx->freeze();

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

        ctx->freeze();

        gpu_memset(gpu0_eid3, 7890, 1);

        volatile int *ptr = (volatile int *)gpu1_eid5->href();
        while (*ptr != 1234) {
        }

        gpu_memset(gpu0_eid4, 3456, 1);

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

// Test accessing remote GPU's memory space after the context is freezed.
unittest::State test_gpu_mgr_remote_lazy_import()
{
    int pid0 = ark::utils::proc_spawn([] {
        unittest::Timeout timeout{10};
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

        mgr->destroy_context(ctx);
        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = ark::utils::proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_gpu_mgr_basic);
    UNITTEST(test_gpu_mgr_remote);
    UNITTEST(test_gpu_mgr_remote_lazy_import);
    return 0;
}
