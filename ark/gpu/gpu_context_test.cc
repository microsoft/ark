// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_context.h"

#include <numeric>

#include "gpu/gpu_mgr.h"
#include "include/ark.h"
#include "unittest/unittest_utils.h"

// Test accessing remote GPU's memory space.
ark::unittest::State test_gpu_context_remote() {
    int pid0 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        std::shared_ptr<ark::GpuContext> ctx = ark::GpuContext::get_context(0, 2);
        std::shared_ptr<ark::GpuBuffer> gpu0_eid3 = ctx->allocate_buffer(sizeof(int));
        std::shared_ptr<ark::GpuBuffer> gpu0_eid4 = ctx->allocate_buffer(sizeof(int));
        ctx->export_buffer(gpu0_eid3, 0, 3);
        ctx->export_buffer(gpu0_eid4, 0, 4);

        std::shared_ptr<ark::GpuBuffer> gpu1_eid5 = ctx->import_buffer(sizeof(int), 1, 5);
        std::shared_ptr<ark::GpuBuffer> gpu1_eid6 = ctx->import_buffer(sizeof(int), 1, 6);
        ctx->freeze(true);

        volatile int *ptr = (volatile int *)gpu0_eid3->ref();
        while (*ptr != 7890) {
        }

        ctx->memset(gpu1_eid5, 0, 1234, 1);
        ptr = (volatile int *)gpu0_eid4->ref();
        while (*ptr != 3456) {
        }

        ctx->memset(gpu1_eid6, 0, 5678, 1);
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        std::shared_ptr<ark::GpuContext> ctx = ark::GpuContext::get_context(1, 2);

        std::shared_ptr<ark::GpuBuffer> gpu1_eid5 = ctx->allocate_buffer(sizeof(int));
        std::shared_ptr<ark::GpuBuffer> gpu1_eid6 = ctx->allocate_buffer(sizeof(int));
        ctx->export_buffer(gpu1_eid5, 0, 5);
        ctx->export_buffer(gpu1_eid6, 0, 6);

        std::shared_ptr<ark::GpuBuffer> gpu0_eid3 = ctx->import_buffer(sizeof(int), 0, 3);
        std::shared_ptr<ark::GpuBuffer> gpu0_eid4 = ctx->import_buffer(sizeof(int), 0, 4);
        ctx->freeze(true);

        ctx->memset(gpu0_eid3, 0, 7890, 1);
        volatile int *ptr = (volatile int *)gpu1_eid5->ref();
        while (*ptr != 1234) {
        }

        ctx->memset(gpu0_eid4, 0, 3456, 1);
        ptr = (volatile int *)gpu1_eid6->ref();
        while (*ptr != 5678) {
        }

        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid1, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gpu_context_remote);
    return 0;
}
