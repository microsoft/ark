// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched.h"

#include "include/ark.h"
#include "logging.h"
#include "ops/ops_test_common.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sched_many_comm_ops() {
    constexpr int num_gpus = 4;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, num_gpus]() {
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m{gpu_id};

            for (int i = 0; i < 100; ++i) {
                ark::Tensor *data = m.tensor(ark::Dims(4096), ark::FP16);
                m.all_gather(data, gpu_id, num_gpus);
            }

            ark::Executor exe{gpu_id, num_gpus, m, "test_sched_many_comm_ops"};
            exe.compile();
            exe.launch();
            exe.run(3);
            exe.stop();
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_mixed_precision() {
    ark::Model m;
    ark::Tensor *x0 = m.tensor({2, 128, 128}, ark::FP16);
    ark::Tensor *x1 = m.scale(x0, 0.7);
    ark::Tensor *x2 = m.cast(x1, ark::FP32);
    ark::Tensor *x3 = m.tensor({2, 128, 128}, ark::FP32);
    m.matmul(x2, x3);

    ark::Executor exe{0, 1, m, "sched_mixed_precision"};
    exe.compile();
    exe.launch();
    exe.run(3);
    exe.stop();

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_parallel_matmul() {
    ark::Model m;
    ark::Tensor *t0 = m.tensor({256, 8192}, ark::FP16);
    ark::Tensor *t1 = m.tensor({8192, 8192}, ark::FP16);
    auto shards = m.sharding(t0, 0, 128);

    m.matmul(t0, t1);
    m.matmul(shards[0], t1);

    ark::Executor exe{0, 1, m, "sched_parallel_matmul"};
    exe.compile();
    exe.launch();
    exe.run(3);
    exe.stop();

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_graph_opt() {
    ark::Model m;
    ark::Tensor *ones = m.tensor({128, 8192}, ark::FP32);
    ark::Tensor *ppp_ones = m.scale(ones, 0.001);
    ark::Tensor *w = m.tensor({8192, 256}, ark::FP32);

    ark::Tensor *y = m.matmul(ppp_ones, w);
    ark::Tensor *ones2 = m.tensor({128, 256}, ark::FP32);
    ark::Tensor *y_plus_one = m.add(y, ones2);

    ark::Executor exe{0, 1, m, "sched_graph_opt"};
    exe.compile();

    std::vector<float> ones_data(ones->shape.size(), 1.0f);
    std::vector<float> ones2_data(ones2->shape.size(), 1.0f);
    std::vector<float> w_data(w->shape.size(), 1.0f);
    ones->write(ones_data.data());
    ones2->write(ones2_data.data());
    w->write(w_data.data());

    exe.launch();
    exe.run(1);
    exe.stop();

    std::vector<float> output_y(y->shape.size());
    y->read(output_y.data());

    for (float v : output_y) {
        UNITTEST_EQ(int(v * 100), 819);
    }

    std::vector<float> output_y_plus_one(y_plus_one->shape.size());
    y_plus_one->read(output_y_plus_one.data());

    for (float v : output_y_plus_one) {
        UNITTEST_EQ(int(v * 100), 919);
    }

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    // UNITTEST(test_sched_many_comm_ops);
    UNITTEST(test_sched_mixed_precision);
    UNITTEST(test_sched_parallel_matmul);
    UNITTEST(test_sched_graph_opt);
    return 0;
}
