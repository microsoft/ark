// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"
#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_ops_sharding_model() {
    // OpNode graph:
    //
    //   ReluOp --+
    //            |
    //   ReluOp --+
    //            |
    //   ReluOp --+--> ReluOp
    //

    ark::Model model;
    ark::Tensor t0 = model.tensor({3}, ark::FP32);

    std::vector<ark::Tensor> vec = model.sharding(t0, 0, 1);
    UNITTEST_EQ(vec.size(), 3);

    ark::Tensor t1 = vec[0];
    ark::Tensor t2 = vec[1];
    ark::Tensor t3 = vec[2];

    ark::Tensor r0 = model.relu(t1);
    ark::Tensor r1 = model.relu(t2);
    ark::Tensor r2 = model.relu(t3);

    ark::Tensor t4 = model.identity(t0, {r0, r1, r2});

    ark::Tensor t5 = model.relu(t4);
    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], r0.ref());
    UNITTEST_EQ(nodes[0]->producers.size(), 0);
    UNITTEST_EQ(nodes[0]->consumers.size(), 1);

    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], r1.ref());
    UNITTEST_EQ(nodes[1]->producers.size(), 0);
    UNITTEST_EQ(nodes[1]->consumers.size(), 1);

    UNITTEST_EQ(nodes[2]->op->result_tensors()[0], r2.ref());
    UNITTEST_EQ(nodes[2]->producers.size(), 0);
    UNITTEST_EQ(nodes[2]->consumers.size(), 1);

    UNITTEST_EQ(nodes[3]->op->result_tensors()[0], t5.ref());
    UNITTEST_EQ(nodes[3]->producers.size(), 3);
    UNITTEST_EQ(nodes[3]->consumers.size(), 0);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_ops_sharding_model);
    return 0;
}
