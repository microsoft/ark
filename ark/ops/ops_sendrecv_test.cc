// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <numeric>

#include "env.h"
#include "include/ark.h"
#include "ipc/ipc_coll.h"
#include "logging.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

void test_sendrecv_internal() {
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            //
            ark::Model model{gpu_id};
            ark::Tensor *tns_x;
            if (gpu_id == 0) {
                tns_x = model.tensor({1024}, ark::FP16);
                tns_x = model.send(tns_x, 0, 1, tns_x->shape_bytes());
                model.send_done(tns_x, 0, 1);
            }
            if (gpu_id == 1) {
                tns_x = model.recv(0, 0, /*bytes=*/2048);
            }

            ark::Executor exe{gpu_id, 2, model, "test_sendrecv"};
            exe.compile();

            if (gpu_id == 0) {
                std::vector<ark::half_t> data(1024);
                std::iota(data.begin(), data.end(), 1.0f);
                tns_x->write(data.data());
            }
            exe.launch();
            exe.run(1);
            exe.stop();

            int tmp[2];
            ark::IpcAllGather barrier{"test_sendrecv_barrier", gpu_id, 2, tmp,
                                      sizeof(int)};
            barrier.sync();

            if (gpu_id == 1) {
                std::vector<ark::half_t> data(1024);
                tns_x->read(data.data());
                for (int i = 0; i < 1024; ++i) {
                    UNITTEST_EQ(data[i], ark::half_t(i + 1));
                }
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
}

ark::unittest::State test_device_sync() {
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            ark::Model model{gpu_id};
            ark::Tensor *tns = model.tensor({1}, ark::FP16);
            model.device_sync(tns, 2);
            ark::Executor exe{gpu_id, 2, model, "test_device_sync"};
            exe.compile();

            exe.launch();
            exe.run(1);
            exe.stop();
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sendrecv() {
    test_sendrecv_internal();
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_sendrecv);
    UNITTEST(test_device_sync);
    return ark::unittest::SUCCESS;
}
