// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "logging.h"
#include "ops_test_common.h"
#include "random.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sendrecv_mm_copy_internal(ark::DimType mat_length) {
    ark::srand();

    ark::DimType mat_size = mat_length * mat_length;
    std::vector<ark::half_t> send_data(mat_size);
    std::generate(send_data.begin(), send_data.end(),
                  []() { return ark::rand<ark::half_t>(-5.0f, 5.0f); });

    ark::unittest::spawn_process([&]() {
        ark::Model m;
        ark::Tensor *data = m.tensor({mat_length, mat_length}, ark::FP16);
        m.send_mm(data, 0, 1, 0);

        ark::Executor exe{0, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        data->write(send_data.data());
        exe.launch();
        exe.run(1);
        exe.stop();

        return ark::unittest::SUCCESS;
    });

    ark::unittest::spawn_process([&]() {
        ark::Model m;
        ark::Tensor *recvbuf = m.tensor({mat_length, mat_length}, ark::FP16);
        m.recv_mm(recvbuf, 0, 0, 0);

        ark::Executor exe{1, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        exe.launch();
        exe.run(1);
        exe.stop();

        std::vector<ark::half_t> recv_data(mat_size, ark::half_t(0.0f));
        recvbuf->read(recv_data.data());

        for (int i = 0; i < mat_size; i++) {
            if (recv_data[i] != send_data[i]) {
                UNITTEST_LOG("error at ", i,
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
    ark::DimType mat_length) {
    ark::srand();

    ark::DimType mat_size = mat_length * mat_length;
    std::vector<ark::half_t> send_data_0(mat_size);
    std::vector<ark::half_t> send_data_1(mat_size);
    std::generate(send_data_0.begin(), send_data_0.end(),
                  []() { return ark::rand<ark::half_t>(-5.0f, 5.0f); });
    std::generate(send_data_1.begin(), send_data_1.end(),
                  []() { return ark::rand<ark::half_t>(-5.0f, 5.0f); });

    ark::unittest::spawn_process([&]() {
        ark::Model m;

        ark::Tensor *data = m.tensor({mat_length, mat_length}, ark::FP16);
        m.send_mm(data, 0, 1, 0);

        ark::Tensor *recvbuf = m.tensor({mat_length, mat_length}, ark::FP16);
        m.recv_mm(recvbuf, 1, 1, 0);

        ark::Executor exe{0, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        data->write(send_data_0.data());
        exe.launch();
        exe.run(1);
        exe.stop();

        std::vector<ark::half_t> recv_data(mat_size, ark::half_t(0.0f));
        recvbuf->read(recv_data.data());

        for (int i = 0; i < mat_size; i++) {
            if (recv_data[i] != send_data_1[i]) {
                UNITTEST_LOG("error at ", i,
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

        ark::Executor exe{1, 2, m, "test_sendrecv_mm_copy"};
        exe.compile();
        data->write(send_data_1.data());
        exe.launch();
        exe.run(1);
        exe.stop();

        std::vector<ark::half_t> recv_data(mat_size, ark::half_t(0.0f));
        recvbuf->read(recv_data.data());

        for (int i = 0; i < mat_size; i++) {
            if (recv_data[i] != send_data_0[i]) {
                UNITTEST_LOG("error at ", i,
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

ark::unittest::State test_sendrecv_mm_4gpus() {
    // the four gpus send recv data in a ring, gpu0->gpu1->gpu2->gpu3->gpu0
    const int gpu_num = 4;
    ark::DimType mat_length = 64;
    ark::DimType mat_size = mat_length * mat_length;

    std::vector<std::vector<ark::half_t>> send_data(gpu_num);
    for (int i = 0; i < gpu_num; ++i) {
        send_data[i].resize(mat_size);
        std::generate(send_data[i].begin(), send_data[i].end(),
                      []() { return ark::rand<ark::half_t>(-5.0f, 5.0f); });
    }

    for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
        ark::unittest::spawn_process([&]() {
            ark::Model m;
            ark::Tensor *data = m.tensor({mat_length, mat_length}, ark::FP16);
            m.send_mm(data, (gpu_id + 1) % gpu_num, (gpu_id + 1) % gpu_num);

            ark::Tensor *recvbuf =
                m.tensor({mat_length, mat_length}, ark::FP16);
            m.recv_mm(recvbuf, gpu_id, (gpu_id - 1 + gpu_num) % gpu_num);

            ark::Executor exe{gpu_id, gpu_num, m, "test_sendrecv_mm_copy"};
            exe.compile();
            data->write(send_data[gpu_id].data());
            exe.launch();
            exe.run(1);
            exe.stop();

            std::vector<ark::half_t> recv_data(mat_size, ark::half_t(0.0f));
            recvbuf->read(recv_data.data());

            auto &gt = send_data[(gpu_id - 1 + gpu_num) % gpu_num];
            for (int i = 0; i < mat_size; i++) {
                if (recv_data[i] != gt[i]) {
                    UNITTEST_LOG("error at ", i,
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

ark::unittest::State test_sendrecv_mm_copy() {
    test_sendrecv_mm_copy_internal(64);
    test_sendrecv_mm_copy_internal(2048);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sendrecv_mm_copy_bidir() {
    test_sendrecv_mm_copy_bidir_internal(64);
    test_sendrecv_mm_copy_bidir_internal(2048);

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_sendrecv_mm_copy);
    UNITTEST(test_sendrecv_mm_copy_bidir);
    // UNITTEST(test_sendrecv_mm_4gpus);
    return 0;
}
