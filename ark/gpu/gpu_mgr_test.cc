// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mgr.h"

#include <numeric>

#include "gpu/gpu_mgr_v2.h"
#include "include/ark.h"
#include "unittest/unittest_utils.h"

// Test initializing and destroying GpuMgr and GpuMgrCtx.
ark::unittest::State test_gpu_mgr_basic() {
    int pid = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        ark::GpuMgr *mgr_a = ark::get_gpu_mgr(0);
        ark::GpuMgr *mgr_b = ark::get_gpu_mgr(0);

        UNITTEST_EQ(mgr_a, mgr_b);

        ark::GpuMgr *mgr_c = ark::get_gpu_mgr(1);

        UNITTEST_NE(mgr_a, mgr_c);

        ark::GpuMgrCtx *ctx_a = mgr_a->create_context("test_a", 0, 1);
        UNITTEST_NE(ctx_a, (ark::GpuMgrCtx *)nullptr);
        UNITTEST_EQ(ctx_a->get_name(), "test_a");
        UNITTEST_EQ(ctx_a->get_rank(), 0);
        UNITTEST_EQ(ctx_a->get_world_size(), 1);
        UNITTEST_EQ(ctx_a->get_gpu_id(), 0);

        mgr_a->destroy_context(ctx_a);

        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gpu_mgr_mem_alloc() {
    int pid = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        ark::GpuMgr *mgr = ark::get_gpu_mgr(0);

        ark::GpuMgrCtx *ctx = mgr->create_context("test", 0, 1);
        UNITTEST_NE(ctx, (ark::GpuMgrCtx *)nullptr);

        ark::GpuBuf *buf0 = ctx->mem_alloc(sizeof(int));
        ark::GpuBuf *buf1 = ctx->mem_alloc(sizeof(int));

        UNITTEST_TRUE(ctx->get_total_bytes() >= 2 * sizeof(int));

        ctx->freeze();

        int buf0_data = 7;
        int buf1_data = 8;
        ark::gpu_memcpy(buf0, 0, &buf0_data, 0, sizeof(int));
        ark::gpu_memcpy(buf1, 0, &buf1_data, 0, sizeof(int));

        int buf0_data2 = 0;
        int buf1_data2 = 0;

        ark::gpu_memcpy(&buf0_data2, 0, buf0, 0, sizeof(int));
        ark::gpu_memcpy(&buf1_data2, 0, buf1, 0, sizeof(int));

        UNITTEST_EQ(buf0_data2, 7);
        UNITTEST_EQ(buf1_data2, 8);

        ark::gpu_memcpy(buf0, 0, buf1, 0, sizeof(int));

        ark::gpu_memcpy(&buf0_data2, 0, buf0, 0, sizeof(int));
        UNITTEST_EQ(buf0_data2, 8);

        mgr->destroy_context(ctx);

        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gpu_mgr_mem_free() {
    int pid = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        ark::GpuMgr *mgr = ark::get_gpu_mgr(0);

        ark::GpuMgrCtx *ctx = mgr->create_context("test", 0, 1);
        UNITTEST_NE(ctx, (ark::GpuMgrCtx *)nullptr);

        ark::GpuBuf *buf0 = ctx->mem_alloc(sizeof(int));

        // This does not mean to free the memory, but means that the following
        // `mem_alloc()` can reuse the memory of `buf0`.
        ctx->mem_free(buf0);

        // This should reuse the memory of `buf0`.
        ark::GpuBuf *buf1 = ctx->mem_alloc(sizeof(int));

        ctx->freeze();

        UNITTEST_EQ(buf0->ref(), buf1->ref());
        UNITTEST_EQ(buf0->href(), buf1->href());
        UNITTEST_EQ(buf0->pref(), buf1->pref());

        int buf0_data = 9;
        ark::gpu_memcpy(buf0, 0, &buf0_data, 0, sizeof(int));

        int buf1_data = 0;

        ark::gpu_memcpy(&buf1_data, 0, buf1, 0, sizeof(int));

        UNITTEST_EQ(buf1_data, 9);

        mgr->destroy_context(ctx);

        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

// Test accessing remote GPU's memory space.
ark::unittest::State test_gpu_mgr_remote() {
    int pid0 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
        ark::GpuMgrCtx *ctx = mgr->create_context("test", 0, 2);

        ark::GpuBuf *gpu0_eid3 = ctx->mem_alloc(sizeof(int));
        ark::GpuBuf *gpu0_eid4 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu0_eid3, 0, 3);
        ctx->mem_export(gpu0_eid4, 0, 4);

        ark::GpuBuf *gpu1_eid5 = ctx->mem_import(sizeof(int), 5, 1);
        ark::GpuBuf *gpu1_eid6 = ctx->mem_import(sizeof(int), 6, 1);

        ctx->freeze(true);

        volatile int *ptr = (volatile int *)gpu0_eid3->href();
        while (*ptr != 7890) {
        }

        ark::gpu_memset(gpu1_eid5, 0, 1234, 1);

        ptr = (volatile int *)gpu0_eid4->href();
        while (*ptr != 3456) {
        }

        ark::gpu_memset(gpu1_eid6, 0, 5678, 1);

        mgr->destroy_context(ctx);
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        ark::GpuMgr *mgr = ark::get_gpu_mgr(1);
        ark::GpuMgrCtx *ctx = mgr->create_context("test", 1, 2);

        ark::GpuBuf *gpu1_eid5 = ctx->mem_alloc(sizeof(int));
        ark::GpuBuf *gpu1_eid6 = ctx->mem_alloc(sizeof(int));
        ctx->mem_export(gpu1_eid5, 0, 5);
        ctx->mem_export(gpu1_eid6, 0, 6);

        ark::GpuBuf *gpu0_eid3 = ctx->mem_import(sizeof(int), 3, 0);
        ark::GpuBuf *gpu0_eid4 = ctx->mem_import(sizeof(int), 4, 0);

        ctx->freeze(true);

        ark::gpu_memset(gpu0_eid3, 0, 7890, 1);

        volatile int *ptr = (volatile int *)gpu1_eid5->href();
        while (*ptr != 1234) {
        }

        ark::gpu_memset(gpu0_eid4, 0, 3456, 1);

        ptr = (volatile int *)gpu1_eid6->href();
        while (*ptr != 5678) {
        }

        mgr->destroy_context(ctx);
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid1, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gpu_mgr_2() {
    int pid = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        ark::GpuMgrV2 gmgr(0);
        auto mem = gmgr.malloc(1024);
        std::vector<int> data(1024 / sizeof(int));
        std::iota(data.begin(), data.end(), 0);
        mem->from_host(data);

        std::vector<int> copy;
        mem->to_host(copy);

        for (size_t i = 0; i < data.size(); ++i) {
            UNITTEST_EQ(data[i], copy[i]);
        }

        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gpu_mgr_basic);
    UNITTEST(test_gpu_mgr_mem_alloc);
    UNITTEST(test_gpu_mgr_mem_free);
    UNITTEST(test_gpu_mgr_remote);
    UNITTEST(test_gpu_mgr_2);
    return 0;
}
