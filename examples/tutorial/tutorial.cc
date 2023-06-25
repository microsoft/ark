// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include "ark_utils.h"

using namespace std;
using namespace ark;

int main(int argc, const char **argv)
{
    int N = 1, C = 3, H = 256, W = 256;
    Model model;
    Tensor input = model.tensor({N, C, H, W}, FP16);
    Tensor other = model.tensor({N, C, H, W}, FP16);
    Tensor output = model.add(input, other);
    Executor exe{0, 0, 1, model, "test_add"};
    exe.compile();
    exe.launch();
    exe.run(1);
    exe.stop();
}
