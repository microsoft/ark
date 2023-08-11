// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

using namespace std;

void test_sendrecv_mscclpp_internal()
{
    for (int gpu_id = 0; gpu_id < 2; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id]() {
            //
            int len = 1024;
            ark::Model model{gpu_id};
            ark::Tensor *tns_x = model.tensor({len}, ark::FP16);
            auto data_x = ark::utils::range_halfs(len, 1.0f);
            if (gpu_id == 0) {
                model.send_mscclpp(tns_x, 0, 1, len * sizeof(ark::half_t));
                model.send_done_mscclpp(tns_x, 1);
            }
            if (gpu_id == 1) {
                model.recv_mscclpp(tns_x, 0, 0, len * sizeof(ark::half_t));
            }

            ark::Executor exe{gpu_id, gpu_id, 2, model,
                              "test_sendrecv_mscclpp"};
            exe.compile();
            if (gpu_id == 0) {
                tns_x->write(data_x.get());
            }

            exe.launch();
            exe.run(1);
            exe.stop();
            if (gpu_id == 1) {
                // Copy results of the loop kernel routine into CPU memory.
                void *res = malloc(len * sizeof(ark::half_t));
                tns_x->read(res);

                auto p = ark::utils::cmp_matrix((ark::half_t *)data_x.get(),
                                                (ark::half_t *)res, 1, 1, len);
                float max_err = p.second;
                LOG(ark::INFO, "sendrecv ,bs=", len, setprecision(4), " mse ",
                    p.first, " max_err ", max_err * 100, "%");
            }
            return ark::unittest::SUCCESS;
        });
    }

    ark::unittest::wait_all_processes();
}

ark::unittest::State test_sendrecv_mscclpp()
{
    test_sendrecv_mscclpp_internal();
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sendrecv_mscclpp);
    return ark::unittest::SUCCESS;
}
