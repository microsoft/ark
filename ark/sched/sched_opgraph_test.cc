// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include "logging.h"
#include "sched_opgraph.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sched_opgraph()
{
    ark::Model model;
    ark::Tensor *t0 = model.tensor({1}, ark::FP32);
    ark::Tensor *t1 = model.tensor({1}, ark::FP32);
    ark::Tensor *t2 = model.add(t0, t1);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sched_opgraph);
    return 0;
}
