// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sendrecv_mm_copy_internal(ark::DimType mat_length)
{
    ark::srand();

    ark::DimType mat_size = mat_length * mat_length;
    auto send_data = ark::utils::rand_halfs(mat_size, 5.0f);

    ark::unittest::spawn_process([&]() {
        ark::Model m;
        ark::Tensor *data = m.tensor({mat_length, mat_length}, ark::FP16);
        m.send_mm(data, 0, 1, 0);

        ark::Executor exe{0, 0, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        exe.tensor_memcpy(data, send_data.get(),
                          mat_size * sizeof(ark::half_t));
        exe.launch();
        exe.run(1);
        exe.stop();

        return ark::unittest::SUCCESS;
    });

    ark::unittest::spawn_process([&]() {
        ark::Model m;
        ark::Tensor *recvbuf = m.tensor({mat_length, mat_length}, ark::FP16);
        m.recv_mm(recvbuf, 0, 0, 0);

        ark::Executor exe{1, 1, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        exe.launch();
        exe.run(1);
        exe.stop();

        auto recv_data = ark::utils::zeros<ark::half_t>(mat_size);
        exe.tensor_memcpy(recv_data.get(), recvbuf,
                          mat_size * sizeof(ark::half_t));

        for (int i = 0; i < mat_size; i++) {
            if (recv_data[i] != send_data[i]) {
                LOG(ark::INFO, "error at ", i,
                    ": recv_data=", float(recv_data[i]),
                    "send_data=", float(send_data[i]));
                return ark::unittest::FAILURE;
            }
        }
        return ark::unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sendrecv_mm_copy_bidir_internal(
    ark::DimType mat_length)
{
    ark::srand();

    ark::DimType mat_size = mat_length * mat_length;
    auto send_data_0 = ark::utils::rand_halfs(mat_size, 5.0f);
    auto send_data_1 = ark::utils::rand_halfs(mat_size, 5.0f);

    ark::unittest::spawn_process([&]() {
        ark::Model m;

        ark::Tensor *data = m.tensor({mat_length, mat_length}, ark::FP16);
        m.send_mm(data, 0, 1, 0);

        ark::Tensor *recvbuf = m.tensor({mat_length, mat_length}, ark::FP16);
        m.recv_mm(recvbuf, 1, 1, 0);

        ark::Executor exe{0, 0, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        exe.tensor_memcpy(data, send_data_0.get(),
                          mat_size * sizeof(ark::half_t));
        exe.launch();
        exe.run(1);
        exe.stop();

        auto recv_data = ark::utils::zeros<ark::half_t>(mat_size);
        exe.tensor_memcpy(recv_data.get(), recvbuf,
                          mat_size * sizeof(ark::half_t));

        for (int i = 0; i < mat_size; i++) {
            if (recv_data[i] != send_data_1[i]) {
                LOG(ark::INFO, "error at ", i,
                    ": recv_data=", float(recv_data[i]),
                    "send_data=", float(send_data_1[i]));
                return ark::unittest::FAILURE;
            }
        }
        return ark::unittest::SUCCESS;
    });

    ark::unittest::spawn_process([&]() {
        ark::Model m;
        ark::Tensor *data = m.tensor({mat_length, mat_length}, ark::FP16);
        m.send_mm(data, 1, 0, 0);

        ark::Tensor *recvbuf = m.tensor({mat_length, mat_length}, ark::FP16);
        m.recv_mm(recvbuf, 0, 0, 0);

        ark::Executor exe{1, 1, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        exe.tensor_memcpy(data, send_data_1.get(),
                          mat_size * sizeof(ark::half_t));
        exe.launch();
        exe.run(1);
        exe.stop();

        auto recv_data = ark::utils::zeros<ark::half_t>(mat_size);
        exe.tensor_memcpy(recv_data.get(), recvbuf,
                          mat_size * sizeof(ark::half_t));

        for (int i = 0; i < mat_size; i++) {
            if (recv_data[i] != send_data_0[i]) {
                LOG(ark::INFO, "error at ", i,
                    ": recv_data=", float(recv_data[i]),
                    "send_data=", float(send_data_0[i]));
                return ark::unittest::FAILURE;
            }
        }
        return ark::unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sendrecv_mm_4gpus()
{
    // int iter = ITER;
    // // the four gpus send recv data in a ring, gpu0->gpu1->gpu2->gpu3->gpu0
    // const int gpu_num = 4;
    // const int mat_length = 64;
    // const int mat_size = mat_length * mat_length * sizeof(ark::half_t);
    // char *send_data[gpu_num];
    // for (int i = 0; i < gpu_num; i++) {
    //     send_data[i] = new char[mat_size];
    //     for (int j = 0; j < mat_size; j++)
    //         send_data[i][j] = std::rand() % 256;
    // }
    // for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    //     ark::unittest::spawn_process([=]() {
    //         Model m;
    //         Tensor *copy_data = m.tensor({mat_length, mat_length}, FP16);
    //         m.send_mm(copy_data, (gpu_id + 1) % gpu_num,
    //                   (gpu_id + 1) % gpu_num);
    //         Tensor *recvbuf = m.tensor({mat_length, mat_length}, FP16);
    //         m.recv_mm(recvbuf, gpu_id, (gpu_id - 1 + gpu_num) % gpu_num);
    //         GpuMgr *mgr = get_gpu_mgr(gpu_id);
    //         const GpuInfo &ginfo = mgr->get_gpu_info();
    //         ark::SimpleScheduler sched{m, gpu_id, gpu_id, gpu_num, 8};
    //         GpuMgrCtx *ctx = sched.create_context("test_sendrecv_mm_copy");

    //         ark::GpuBuf *input_data = sched.get_gpu_buf(copy_data);
    //         ark::gpu_memcpy(input_data, send_data[gpu_id], mat_size);

    //         CULOG(cuCtxSynchronize());
    //         sched.schedule();
    //         auto codes = sched.gen_code();

    //         GpuLoopKernel glk{"test_sendrecv_mm_copy",
    //                           codes,
    //                           (unsigned int)ginfo.num_sm,
    //                           8,
    //                           (unsigned int)ginfo.smem_block_total,
    //                           "",
    //                           ctx};
    //         // cout << glk.get_codes()[0] << endl;
    //         glk.compile(ginfo);
    //         glk.load();
    //         GpuStream stream = ctx->create_stream();
    //         GpuState ret = glk.launch(stream, false);
    //         UNITTEST_EQ(ret, 0);
    //         glk.run(iter);
    //         glk.stop();

    //         ark::GpuBuf *output_data;
    //         output_data = sched.get_gpu_buf(recvbuf);
    //         char *output = new char[mat_size];
    //         ark::gpu_memcpy(output, output_data, mat_size);
    //         for (int i = 0; i < mat_size; i++) {
    //             if (output[i] !=
    //                 send_data[(gpu_id - 1 + gpu_num) % gpu_num][i]) {
    //                 LOG(INFO, "error at", i, output[i],
    //                     send_data[(gpu_id - 1 + gpu_num) % gpu_num][i]);
    //                 return ark::unittest::FAILURE;
    //             }
    //         }
    //         return ark::unittest::SUCCESS;
    //     });
    // }

    // ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
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
