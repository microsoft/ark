// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mgr.h"

#include <numeric>

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

int main() {
    ark::init();
    UNITTEST(test_gpu_mgr_basic);
    UNITTEST(test_gpu_mgr_mem_alloc);
    UNITTEST(test_gpu_mgr_mem_free);
    return 0;
}
