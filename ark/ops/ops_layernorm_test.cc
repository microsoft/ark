// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "logging.h"
#include "ops_test_common.h"
#include "random.h"
#include "unittest/unittest_utils.h"

//
void test_layernorm_internal(unsigned int n, unsigned int m, unsigned int k) {
    size_t buf_x_sz = (size_t)m * (size_t)n * (size_t)k * sizeof(ark::half_t);
    size_t buf_y_sz = (size_t)m * (size_t)n * sizeof(ark::half_t);

    // Set data.
    ark::srand();
    std::vector<ark::half_t> data_a(buf_x_sz / sizeof(ark::half_t));
    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = ark::rand<ark::half_t>(-0.01, 0.01);
    }

    // Copy the ground truth results into CPU memory.
    void *gt = std::malloc(buf_y_sz);
    UNITTEST_NE(gt, (void *)nullptr);

    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({m, n, k}, ark::FP32);
    /* ark::Tensor *tns_y = */ model.layernorm(tns_x);

    //
    ark::Executor exe{0, 1, model, "test_layernorm"};
    exe.compile();

    // Set data.
    tns_x->write(data_a.data());

    exe.launch();
    exe.run(1);
    exe.stop();
}

ark::unittest::State test_layernorm() {
    test_layernorm_internal(1, 64, 4);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_layernorm);
    return ark::unittest::SUCCESS;
}
