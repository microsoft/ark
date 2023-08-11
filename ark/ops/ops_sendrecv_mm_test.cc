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
        data->write(send_data.get());
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
        recvbuf->read(recv_data.get());

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
        data->write(send_data_0.get());
        exe.launch();
        exe.run(1);
        exe.stop();

        auto recv_data = ark::utils::zeros<ark::half_t>(mat_size);
        recvbuf->read(recv_data.get());

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
        data->write(send_data_1.get());
        exe.launch();
        exe.run(1);
        exe.stop();

        auto recv_data = ark::utils::zeros<ark::half_t>(mat_size);
        recvbuf->read(recv_data.get());

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
    // the four gpus send recv data in a ring, gpu0->gpu1->gpu2->gpu3->gpu0
    const int gpu_num = 4;
    ark::DimType mat_length = 64;
    ark::DimType mat_size = mat_length * mat_length;

    std::unique_ptr<ark::half_t[]> send_data[gpu_num];
    for (int i = 0; i < gpu_num; ++i) {
        send_data[i] = ark::utils::rand_halfs(mat_size, 5.0f);
    }

    for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
        ark::unittest::spawn_process([&]() {
            ark::Model m;
            ark::Tensor *data = m.tensor({mat_length, mat_length}, ark::FP16);
            m.send_mm(data, (gpu_id + 1) % gpu_num, (gpu_id + 1) % gpu_num);

            ark::Tensor *recvbuf =
                m.tensor({mat_length, mat_length}, ark::FP16);
            m.recv_mm(recvbuf, gpu_id, (gpu_id - 1 + gpu_num) % gpu_num);

            ark::Executor exe{gpu_id, gpu_id, gpu_num, m,
                              "test_sendrecv_mm_copy"};
            exe.compile();
            data->write(send_data[gpu_id].get());
            exe.launch();
            exe.run(1);
            exe.stop();

            auto recv_data = ark::utils::zeros<ark::half_t>(mat_size);
            recvbuf->read(recv_data.get());

            auto &gt = send_data[(gpu_id - 1 + gpu_num) % gpu_num];
            for (int i = 0; i < mat_size; i++) {
                if (recv_data[i] != gt[i]) {
                    LOG(ark::INFO, "error at ", i,
                        ": recv_data=", float(recv_data[i]),
                        "send_data=", float(gt[i]));
                    return ark::unittest::FAILURE;
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
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
