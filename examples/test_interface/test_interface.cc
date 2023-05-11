// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"

using namespace std;

//
void test_scale_internal(unsigned int bs, unsigned int n, unsigned int m,
                         float val = 0.7)
{
    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({bs, n, m}, ark::FP16);
    ark::Tensor *tns_y = model.scale(tns_x, val);

    //
    ark::Executor exe{0, 0, 1, model, "test_scale"};
    exe.compile();

    // Set data.
    exe.launch();
    exe.run(1);
    exe.stop();
}

int main()
{
    // ark::init();
    test_scale_internal(1, 1, 64);

    // return ark::unittest::SUCCESS;
}
