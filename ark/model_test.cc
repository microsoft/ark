// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_simple_mm() {
    // Hidden dimension of the dense layer.
    unsigned int units = 1024;
    // Input dimension of the dense layer.
    unsigned int in_dim = 1024;
    // Extra dimension of the input. CHANNEL=1 for 2D inputs.
    unsigned int channel = 128;
    // Batch size of the input.
    unsigned int batch_size = 1;

    ark::Model m;
    ark::Tensor *input = m.tensor({batch_size, channel, in_dim}, ark::FP16);
    ark::Tensor *weight = m.tensor({in_dim, units}, ark::FP16);
    m.matmul(input, weight);

    UNITTEST_TRUE(m.verify());

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_simple_mm);
    return 0;
}
