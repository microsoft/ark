// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include "ark_utils.h"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace ark;

int main()
{
    ark::Model model;
    // Tensor *input = model.tensor({1, 64}, FP16);
    // Tensor *ground_truth = model.tensor({1, 64}, FP16);
    // Tensor *weight = model.tensor({64, 64}, FP16);
    // Tensor *bias = model.tensor({1, 64}, FP16);
    // Tensor *output1 = model.matmul(input, weight);
    // Tensor *output2 = model.add(output1, bias);

    // Tensor *diff = model.add(output2, ground_truth);
    // Tensor *grad_bias = diff;
    // Tensor *grad_bias_scale = model.tensor(grad_bias->shape,
    // grad_bias->type); model.scale(grad_bias, -0.00001, grad_bias_scale);
    // model.add(bias, grad_bias_scale, model.identity(bias));

    ark::Tensor *tns_a = model.tensor({1, 64, 64}, ark::FP16);
    ark::Tensor *tns_b = model.tensor({1, 64, 64}, ark::FP16);
    model.scale(tns_a, 0.1);
    model.scale(tns_b, 0.2);

    ark::Executor exe{0, 0, 1, model, "test_add"};
    exe.compile();

    exe.launch();
    exe.run(1);
    exe.stop();
}
