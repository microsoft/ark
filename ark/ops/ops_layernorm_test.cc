// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

using namespace std;

//
void test_layernorm_internal(unsigned int n, unsigned int m, unsigned int k)
{
    size_t buf_x_sz = (size_t)m * (size_t)n * (size_t)k * sizeof(ark::half_t);
    size_t buf_y_sz = (size_t)m * (size_t)n * sizeof(ark::half_t);

    // Set data.
    ark::srand();
    auto data_a = ark::utils::rand_halfs(buf_x_sz / sizeof(ark::half_t), 0.01);

    // Copy the ground truth results into CPU memory.
    void *gt = malloc(buf_y_sz);
    UNITTEST_NE(gt, (void *)nullptr);

    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({m, n, k}, ark::FP32);
    /* ark::Tensor *tns_y = */ model.layernorm(tns_x);

    //
    ark::Executor exe{0, 0, 1, model, "test_layernorm"};
    exe.compile();

    // Set data.
    exe.tensor_memcpy(tns_x, data_a.get(), buf_x_sz);

    exe.launch();
    exe.run(1);
    exe.stop();
}

ark::unittest::State test_layernorm()
{
    test_layernorm_internal(1, 64, 4);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_layernorm);
    return ark::unittest::SUCCESS;
}
