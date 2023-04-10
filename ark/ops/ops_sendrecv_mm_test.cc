// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "ark/gpu/gpu_kernel.h"
#include "ark/gpu/gpu_logging.h"
#include "ark/init.h"
#include "ark/logging.h"
#include "ark/ops/ops_test_utils.h"
#include "ark/random.h"
#include "ark/sched/sched.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;
using namespace ark;
#define ITER 1000
ark::unittest::State test_sendrecv_mm_copy_internal(int mat_length)
{
    int mat_size = mat_length * mat_length * sizeof(half_t);
    char *send_data0 = (char *)malloc(mat_size);
    for (int i = 0; i < mat_size; i++)
        send_data0[i] = std::rand() % 256;
    int iter = ITER;
    ark::unittest::spawn_process([=]() {
        Model m;
        // Tensor *data1 = m.tensor({1, mat_length, 1, mat_length},
        // FP16); m.scale(data1, 1.0f);
        Tensor *copy_data = m.tensor({mat_length, mat_length}, FP16);
        m.send_mm(copy_data, 0, 1, 0);
        // m.scale(copy_data, 1.0f);
        GpuMgr *mgr = get_gpu_mgr(0);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::SimpleScheduler sched{0, 0, 2, m, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sendrecv_mm_copy");

        ark::GpuBuf *input_data = sched.get_gpu_buf(copy_data);
        ark::gpu_memcpy(input_data, send_data0, mat_size);

        CULOG(cuCtxSynchronize());
        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_sendrecv_mm_copy",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        // cout << glk.get_codes()[0] << endl;
        glk.compile(ginfo);
        glk.load();
        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(iter);
        glk.stop();
        float time = glk.get_elapsed_msec();
        float bandwidth = (float)mat_size * iter / time / 1000 / 1000;
        LOG(INFO, "test_sendrecv_mm_copy_internal", " matlength: ", mat_length,
            "  bytes: ", mat_size,
            " sender GPU elapsed_time: ", glk.get_elapsed_msec(), "ms",
            " bandwidth: ", bandwidth, "MB/s");
        return ark::unittest::SUCCESS;
    });

    ark::unittest::spawn_process([=]() {
        Model m;
        // Tensor *data1 = m.tensor({1, mat_length, 1, mat_length},
        // FP16); m.scale(data1, 1.0f);
        Tensor *recvbuf = m.tensor({mat_length, mat_length}, FP16);
        m.recv_mm(recvbuf, 0, 0, 0);
        GpuMgr *mgr = get_gpu_mgr(1);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::SimpleScheduler sched{1, 1, 2, m, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sendrecv_mm_copy");

        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_sendrecv_mm_copy",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        // cout << glk.get_codes()[0] << endl;
        glk.compile(ginfo);
        glk.load();

        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(iter);
        glk.stop();

        ark::GpuBuf *output_data;
        output_data = sched.get_gpu_buf(recvbuf);
        char *output = new char[mat_size];
        ark::gpu_memcpy(output, output_data, mat_size);
        for (int i = 0; i < mat_size; i++) {
            if (output[i] != send_data0[i]) {
                LOG(INFO, "error at", i, output[i], send_data0[i]);
                return ark::unittest::FAILURE;
            }
        }
        float time = glk.get_elapsed_msec();
        float bandwidth = (float)mat_size * iter / time / 1000 / 1000;
        LOG(INFO, "test_sendrecv_mm_copy_internal", " matlength: ", mat_length,
            "  bytes: ", mat_size,
            " receiver GPU elapsed_time: ", glk.get_elapsed_msec(), "ms",
            " bandwidth: ", bandwidth, "MB/s");
        return ark::unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return unittest::SUCCESS;
}

ark::unittest::State test_sendrecv_mm_copy_bidir_internal(int mat_length)
{
    int iter = ITER;
    int mat_size = mat_length * mat_length * sizeof(half_t);
    char *send_data0 = (char *)malloc(mat_size);
    for (int i = 0; i < mat_size; i++)
        send_data0[i] = std::rand() % 256;
    char *send_data1 = (char *)malloc(mat_size);
    for (int i = 0; i < mat_size; i++)
        send_data1[i] = std::rand() % 256;
    ark::unittest::spawn_process([=]() {
        Model m;
        Tensor *data1 = m.tensor({mat_length, mat_length}, FP16);
        m.scale(data1, 1.0f);
        Tensor *copy_data = m.tensor({mat_length, mat_length}, FP16);
        m.send_mm(copy_data, 0, 1, 0);
        // m.scale(copy_data, 1.0f);
        Tensor *recvbuf = m.tensor({mat_length, mat_length}, FP16);
        m.recv_mm(recvbuf, 1, 1, 0);
        GpuMgr *mgr = get_gpu_mgr(0);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::SimpleScheduler sched{0, 0, 2, m, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sendrecv_mm_copy");

        ark::GpuBuf *input_data = sched.get_gpu_buf(copy_data);
        ark::gpu_memcpy(input_data, send_data0, mat_size);

        CULOG(cuCtxSynchronize());
        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_sendrecv_mm_copy",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        // cout << glk.get_codes()[0] << endl;
        glk.compile(ginfo);
        glk.load();
        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(iter);
        glk.stop();

        ark::GpuBuf *output_data;
        output_data = sched.get_gpu_buf(recvbuf);
        char *output = new char[mat_size];
        ark::gpu_memcpy(output, output_data, mat_size);
        for (int i = 0; i < mat_size; i++) {
            if (output[i] != send_data1[i]) {
                LOG(INFO, "error at", i, output[i], send_data1[i]);
                return ark::unittest::FAILURE;
            }
        }
        float time = glk.get_elapsed_msec();
        float bandwidth = (float)mat_size * iter / time / 1000 / 1000;
        LOG(INFO, "test_sendrecv_mm_copy_bidir_internal",
            " matlength: ", mat_length, "  bytes: ", mat_size,
            " GPU 0 elapsed_time: ", glk.get_elapsed_msec(), "ms",
            " bandwidth: ", bandwidth, "MB/s");
        return ark::unittest::SUCCESS;
    });

    ark::unittest::spawn_process([=]() {
        Model m;
        Tensor *copy_data = m.tensor({mat_length, mat_length}, FP16);
        m.send_mm(copy_data, 1, 0, 0);

        Tensor *recvbuf = m.tensor({mat_length, mat_length}, FP16);
        m.recv_mm(recvbuf, 0, 0, 0);
        GpuMgr *mgr = get_gpu_mgr(1);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::SimpleScheduler sched{1, 1, 2, m, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sendrecv_mm_copy");

        ark::GpuBuf *input_data = sched.get_gpu_buf(copy_data);
        ark::gpu_memcpy(input_data, send_data1, mat_size);

        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_sendrecv_mm_copy",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        // cout << glk.get_codes()[0] << endl;
        glk.compile(ginfo);
        glk.load();

        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(iter);
        glk.stop();

        ark::GpuBuf *output_data;
        output_data = sched.get_gpu_buf(recvbuf);
        char *output = new char[mat_size];
        ark::gpu_memcpy(output, output_data, mat_size);
        for (int i = 0; i < mat_size; i++) {
            if (output[i] != send_data0[i]) {
                LOG(INFO, "error at", i, output[i], send_data0[i]);
                return ark::unittest::FAILURE;
            }
        }
        float time = glk.get_elapsed_msec();
        float bandwidth = (float)mat_size * iter / time / 1000 / 1000;
        LOG(INFO, "test_sendrecv_mm_copy_bidir_internal",
            " matlength: ", mat_length, "  bytes: ", mat_size,
            " GPU 1 elapsed_time: ", glk.get_elapsed_msec(), "ms",
            " bandwidth: ", bandwidth, "MB/s");
        return ark::unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return unittest::SUCCESS;
}

ark::unittest::State test_sendrecv_mm_4gpus()
{
    int iter = ITER;
    // the four gpus send recv data in a ring, gpu0->gpu1->gpu2->gpu3->gpu0
    const int gpu_num = 4;
    const int mat_length = 64;
    const int mat_size = mat_length * mat_length * sizeof(half_t);
    char *send_data[gpu_num];
    for (int i = 0; i < gpu_num; i++) {
        send_data[i] = new char[mat_size];
        for (int j = 0; j < mat_size; j++)
            send_data[i][j] = std::rand() % 256;
    }
    for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
        ark::unittest::spawn_process([=]() {
            Model m;
            Tensor *copy_data = m.tensor({mat_length, mat_length}, FP16);
            m.send_mm(copy_data, (gpu_id + 1) % gpu_num,
                      (gpu_id + 1) % gpu_num);
            Tensor *recvbuf = m.tensor({mat_length, mat_length}, FP16);
            m.recv_mm(recvbuf, gpu_id, (gpu_id - 1 + gpu_num) % gpu_num);
            GpuMgr *mgr = get_gpu_mgr(gpu_id);
            const GpuInfo &ginfo = mgr->get_gpu_info();
            ark::SimpleScheduler sched{gpu_id, gpu_id, gpu_num, m, 8};
            GpuMgrCtx *ctx = sched.create_context("test_sendrecv_mm_copy");

            ark::GpuBuf *input_data = sched.get_gpu_buf(copy_data);
            ark::gpu_memcpy(input_data, send_data[gpu_id], mat_size);

            CULOG(cuCtxSynchronize());
            auto codes = sched.schedule();
            unsigned int num_depths = sched.get_num_depths();

            GpuLoopKernel glk{"test_sendrecv_mm_copy",
                              codes,
                              (unsigned int)ginfo.num_sm,
                              8,
                              (unsigned int)ginfo.smem_block_total,
                              "",
                              ctx,
                              num_depths};
            // cout << glk.get_codes()[0] << endl;
            glk.compile(ginfo);
            glk.load();
            GpuStream stream = ctx->create_stream();
            GpuState ret = glk.launch(stream, false);
            UNITTEST_EQ(ret, 0);
            glk.run(iter);
            glk.stop();

            ark::GpuBuf *output_data;
            output_data = sched.get_gpu_buf(recvbuf);
            char *output = new char[mat_size];
            ark::gpu_memcpy(output, output_data, mat_size);
            for (int i = 0; i < mat_size; i++) {
                if (output[i] !=
                    send_data[(gpu_id - 1 + gpu_num) % gpu_num][i]) {
                    LOG(INFO, "error at", i, output[i],
                        send_data[(gpu_id - 1 + gpu_num) % gpu_num][i]);
                    return ark::unittest::FAILURE;
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return unittest::SUCCESS;
}

ark::unittest::State test_sendrecv_mm_copy()
{
    test_sendrecv_mm_copy_internal(64);
    test_sendrecv_mm_copy_internal(128);
    test_sendrecv_mm_copy_internal(256);
    test_sendrecv_mm_copy_internal(512);
    test_sendrecv_mm_copy_internal(1024);
    test_sendrecv_mm_copy_internal(2048);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sendrecv_mm_copy_bidir()
{
    test_sendrecv_mm_copy_bidir_internal(64);
    test_sendrecv_mm_copy_bidir_internal(128);
    test_sendrecv_mm_copy_bidir_internal(256);
    test_sendrecv_mm_copy_bidir_internal(512);
    test_sendrecv_mm_copy_bidir_internal(1024);
    test_sendrecv_mm_copy_bidir_internal(2048);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sendrecv_mm_copy);
    UNITTEST(test_sendrecv_mm_copy_bidir);
    UNITTEST(test_sendrecv_mm_4gpus);
    return 0;
}