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
    ark::Tensor *tns_a = model.tensor({1, 64, 64}, ark::FP16);
    ark::Tensor *tns_b = model.tensor({1, 64, 64}, ark::FP16);
    model.add(tns_a, tns_b, tns_a);

    //
    ark::Executor exe{0, 0, 1, model, "test_add"};
    exe.compile();

    exe.launch();
    exe.run(1);
    exe.stop();
}
