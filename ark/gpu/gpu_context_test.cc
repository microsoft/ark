// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_context.h"

#include <numeric>

#include "include/ark.h"
#include "unittest/unittest_utils.h"

// Test initializing and destroying GpuContext
ark::unittest::State test_gpu_context_basic() {
    int pid = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        std::shared_ptr<ark::GpuContext> ctx =
            ark::GpuContext::get_context(0, 1);
        UNITTEST_NE(ctx, nullptr);
        UNITTEST_EQ(ctx->rank(), 0);
        UNITTEST_EQ(ctx->world_size(), 1);
        UNITTEST_EQ(ctx->gpu_id(), 0);

        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gpu_context_buffer_alloc() {
    int pid = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        std::shared_ptr<ark::GpuContext> ctx =
            ark::GpuContext::get_context(0, 1);
        UNITTEST_NE(ctx, nullptr);

        std::shared_ptr<ark::GpuBuffer> buf0 =
            ctx->allocate_buffer(sizeof(int));
        std::shared_ptr<ark::GpuBuffer> buf1 =
            ctx->allocate_buffer(sizeof(int));

        UNITTEST_TRUE(ctx->get_total_bytes() >= 2 * sizeof(int));

        ctx->freeze();

        int buf0_data = 7;
        int buf1_data = 8;
        buf0->memcpy_from(0, &buf0_data, 0, sizeof(int));
        buf1->memcpy_from(0, &buf1_data, 0, sizeof(int));

        int buf0_data2 = 0;
        int buf1_data2 = 0;

        buf0->memcpy_to(&buf0_data2, 0, 0, sizeof(int));
        buf1->memcpy_to(&buf1_data2, 0, 0, sizeof(int));

        UNITTEST_EQ(buf0_data2, 7);
        UNITTEST_EQ(buf1_data2, 8);

        buf0->memcpy_from(0, *buf1, 0, sizeof(int));
        buf0->memcpy_to(&buf0_data2, 0, 0, sizeof(int));
        UNITTEST_EQ(buf0_data2, 8);
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gpu_context_buffer_free() {
    int pid = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        std::shared_ptr<ark::GpuContext> ctx =
            ark::GpuContext::get_context(0, 1);
        UNITTEST_NE(ctx, nullptr);

        std::shared_ptr<ark::GpuBuffer> buf0 =
            ctx->allocate_buffer(sizeof(int));
        // This does not mean to free the buffer, but means that the following
        // `allocate_buffer()` can reuse the memory of `buf0`.
        ctx->free_buffer(buf0);
        // This should reuse the memory of `buf0`.
        std::shared_ptr<ark::GpuBuffer> buf1 =
            ctx->allocate_buffer(sizeof(int));
        ctx->freeze();

        UNITTEST_EQ(buf0->ref(), buf1->ref());

        int buf0_data = 9;
        buf0->memcpy_from(0, &buf0_data, 0, sizeof(int));

        int buf1_data = 0;
        buf1->memcpy_to(&buf1_data, 0, 0, sizeof(int));
        UNITTEST_EQ(buf1_data, 9);

        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

// Test accessing remote GPU's memory space.
ark::unittest::State test_gpu_context_remote() {
    int pid0 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        std::shared_ptr<ark::GpuContext> ctx =
            ark::GpuContext::get_context(0, 2);
        std::shared_ptr<ark::GpuBuffer> gpu0_eid3 =
            ctx->allocate_buffer(sizeof(int));
        std::shared_ptr<ark::GpuBuffer> gpu0_eid4 =
            ctx->allocate_buffer(sizeof(int));
        ctx->export_buffer(gpu0_eid3, 0, 3);
        ctx->export_buffer(gpu0_eid4, 0, 4);

        std::shared_ptr<ark::GpuBuffer> gpu1_eid5 =
            ctx->import_buffer(sizeof(int), 1, 5);
        std::shared_ptr<ark::GpuBuffer> gpu1_eid6 =
            ctx->import_buffer(sizeof(int), 1, 6);
        ctx->freeze(true);

        volatile int *ptr = (volatile int *)gpu0_eid3->ref();
        while (*ptr != 7890) {
        }

        gpu1_eid5->memset_d32(1234, 0, 1);
        ptr = (volatile int *)gpu0_eid4->ref();
        while (*ptr != 3456) {
        }

        gpu1_eid6->memset_d32(5678, 0, 1);
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        std::shared_ptr<ark::GpuContext> ctx =
            ark::GpuContext::get_context(1, 2);

        std::shared_ptr<ark::GpuBuffer> gpu1_eid5 =
            ctx->allocate_buffer(sizeof(int));
        std::shared_ptr<ark::GpuBuffer> gpu1_eid6 =
            ctx->allocate_buffer(sizeof(int));
        ctx->export_buffer(gpu1_eid5, 0, 5);
        ctx->export_buffer(gpu1_eid6, 0, 6);

        std::shared_ptr<ark::GpuBuffer> gpu0_eid3 =
            ctx->import_buffer(sizeof(int), 0, 3);
        std::shared_ptr<ark::GpuBuffer> gpu0_eid4 =
            ctx->import_buffer(sizeof(int), 0, 4);
        ctx->freeze(true);

        gpu0_eid3->memset_d32(7890, 0, 1);
        volatile int *ptr = (volatile int *)gpu1_eid5->ref();
        while (*ptr != 1234) {
        }

        gpu0_eid4->memset_d32(3456, 0, 1);
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
    UNITTEST(test_gpu_context_basic);
    UNITTEST(test_gpu_context_buffer_free);
    UNITTEST(test_gpu_context_buffer_alloc);
    UNITTEST(test_gpu_context_remote);
    return 0;
}
