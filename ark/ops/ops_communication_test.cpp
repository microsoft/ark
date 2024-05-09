// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <numeric>

#include "ark/executor.hpp"
#include "half.h"
#include "ops_test_common.hpp"

ark::unittest::State test_communication_send_recv_unidir() {
    // send from gpu 0 to gpu 1
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            ark::Model model(gpu_id, 2);
            ark::Tensor tns = model.tensor({1024}, ark::FP16);
            if (gpu_id == 0) {
                tns = model.send(tns, 1, 0);
                model.send_done(tns);
            }
            if (gpu_id == 1) {
                tns = model.recv(tns, 0, 0);
            }

            ark::DefaultExecutor exe(model, gpu_id);
            exe.compile();

            if (gpu_id == 0) {
                std::vector<ark::half_t> data(1024);
                std::iota(data.begin(), data.end(), 1.0f);
                exe.tensor_write(tns, data);
            }

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            if (gpu_id == 1) {
                std::vector<ark::half_t> data(1024);
                exe.tensor_read(tns, data);
                for (int i = 0; i < 1024; ++i) {
                    UNITTEST_EQ(data[i], ark::half_t(i + 1));
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();

    // send from gpu 1 to gpu 0
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            ark::Model model(gpu_id, 2);
            ark::Tensor tns = model.tensor({1024}, ark::FP16);
            if (gpu_id == 1) {
                tns = model.send(tns, 0, 0);
                model.send_done(tns);
            }
            if (gpu_id == 0) {
                tns = model.recv(tns, 1, 0);
            }

            ark::DefaultExecutor exe(model, gpu_id);
            exe.compile();

            if (gpu_id == 1) {
                std::vector<ark::half_t> data(1024);
                std::iota(data.begin(), data.end(), 1.0f);
                exe.tensor_write(tns, data);
            }

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            if (gpu_id == 0) {
                std::vector<ark::half_t> data(1024);
                exe.tensor_read(tns, data);
                for (int i = 0; i < 1024; ++i) {
                    UNITTEST_EQ(data[i], ark::half_t(i + 1));
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_communication_send_recv_bidir() {
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            int remote_gpu_id = (gpu_id + 1) % 2;
            int tag = 0;

            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            ark::Tensor tns = model.send(tns_data, remote_gpu_id, tag);
            tns = model.send_done(tns);

            ark::Tensor tns2_data = model.tensor({1024}, ark::FP16);
            // build dependency (send_done --> recv)
            ark::Tensor tns2 = model.identity(tns2_data, {tns});
            tns2 = model.recv(tns2_data, remote_gpu_id, tag);

            ark::DefaultExecutor exe(model, gpu_id);
            exe.compile();

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), ark::half_t(gpu_id + 1));
            exe.tensor_write(tns_data, data);

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            data.clear();
            data.resize(1024);
            exe.tensor_read(tns2_data, data);
            for (int i = 0; i < 1024; ++i) {
                UNITTEST_EQ(data[i], ark::half_t(remote_gpu_id + i + 1));
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();

    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            int remote_gpu_id = (gpu_id + 1) % 2;
            int tag = 0;

            ark::Model model(gpu_id, 2);
            ark::Tensor tns_data = model.tensor({1024}, ark::FP16);
            ark::Tensor tns = model.send(tns_data, remote_gpu_id, tag);
            tns = model.send_done(tns);

            ark::Tensor tns2_data = model.tensor({1024}, ark::FP16);
            // build dependency (send_done --> recv)
            ark::Tensor tns2 = model.identity(tns2_data, {tns});
            tns2 = model.recv(tns2_data, remote_gpu_id, tag);

            ark::Tensor sum = model.add(tns2, tns_data);

            ark::DefaultExecutor exe(model, gpu_id);
            exe.compile();

            std::vector<ark::half_t> data(1024);
            std::iota(data.begin(), data.end(), ark::half_t(gpu_id + 1));
            exe.tensor_write(tns_data, data);

            exe.barrier();

            exe.launch();
            exe.run(1);
            exe.stop();

            exe.barrier();

            data.clear();
            data.resize(1024);
            exe.tensor_read(sum, data);
            for (int i = 0; i < 1024; ++i) {
                UNITTEST_EQ(data[i],
                            ark::half_t(gpu_id + remote_gpu_id + 2 * i + 2));
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_communication_send_recv_unidir);
    UNITTEST(test_communication_send_recv_bidir);
    return ark::unittest::SUCCESS;
}