// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/logging.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

//
void test_gelu_internal(unsigned int bs, unsigned int n, unsigned int m,
                        float val = 0.7)
{
    unsigned int len = bs * m * n;
    // Set data.
    ark::srand();
    auto data_x = ark::utils::rand_halfs(len, 0.01);

    // Copy the ground truth results into CPU memory.
    void *gt = malloc(len * sizeof(ark::half_t));
    UNITTEST_NE(gt, (void *)nullptr);

    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({bs, n, m}, ark::FP16);
    ark::Tensor *tns_y = model.gelu(tns_x);

    //
    ark::Executor exe{0, 0, 1, model, "test_gelu"};
    exe.compile();

    // Set data.
    exe.tensor_memcpy(tns_x, data_x.get(), len * sizeof(ark::half_t));

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    void *res = malloc(len * sizeof(ark::half_t));
    UNITTEST_NE(res, (void *)nullptr);
    exe.tensor_memcpy(res, tns_y, len * sizeof(ark::half_t));

    // Compare results with the ground truth.
    auto p =
        ark::utils::cmp_matrix((ark::half_t *)gt, (ark::half_t *)res, m, n, bs);
    float max_err = p.second;
    LOG(ark::INFO, "gelu:", n, 'x', m, ",bs=", bs, setprecision(4), " mse ",
        p.first, " max_err ", max_err * 100, "%");

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

ark::unittest::State test_gelu()
{
    // test_gelu_internal(1, 1, 64);
    // test_gelu_internal(1, 128, 128);
    // test_gelu_internal(1, 4096, 1024);
    // test_gelu_internal(1, 1024, 4096);
    // test_gelu_internal(2, 1, 64);
    // test_gelu_internal(2, 128, 128);
    // test_gelu_internal(8, 4096, 1024);
    // test_gelu_internal(8, 1024, 4096);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_gelu);
    return ark::unittest::SUCCESS;
}
