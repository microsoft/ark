// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"
#include <cmath>

using namespace std;

float gelu(float x)
{
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

//
void test_gelu_internal(unsigned int bs, unsigned int n, unsigned int m)
{
    unsigned int len = bs * m * n;
    // Set data.
    ark::srand();
    auto data_x = ark::utils::rand_halfs(len, 0.01);

    // Get ground truth
    void *gt = malloc(len * sizeof(ark::half_t));
    UNITTEST_NE(gt, (void *)nullptr);
    for (unsigned int i = 0; i < len; ++i) {
        ((ark::half_t *)gt)[i] = gelu(((ark::half_t *)data_x.get())[i]);
    }
    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({bs, n, m}, ark::FP16);
    ark::Tensor *tns_y = model.gelu(tns_x);

    //
    ark::Executor exe{0, 0, 1, model, "test_gelu"};
    exe.compile();

    // Set data.
    tns_x->write(data_x.get());

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    void *res = malloc(len * sizeof(ark::half_t));
    UNITTEST_NE(res, (void *)nullptr);
    tns_y->read(res);

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
    test_gelu_internal(1, 1, 64);
    test_gelu_internal(1, 64, 64);
    test_gelu_internal(1, 128, 128);
    test_gelu_internal(1, 4096, 1024);
    test_gelu_internal(1, 1024, 4096);
    test_gelu_internal(2, 1, 64);
    test_gelu_internal(2, 128, 128);
    test_gelu_internal(8, 4096, 1024);
    test_gelu_internal(8, 1024, 4096);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_gelu);
    return ark::unittest::SUCCESS;
}
