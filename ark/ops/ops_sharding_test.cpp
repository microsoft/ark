// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"
#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_model_op_sharding() {
    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (Relu,) --+
    //             |
    //   (Relu,) --+
    //             |
    //   (Relu,) --+--> (Relu,)
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
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    auto nodes_iter = nodes.begin();
    auto node = *(nodes_iter++);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r0.ref());
    UNITTEST_EQ(node->producers.size(), 0);
    UNITTEST_EQ(node->consumers.size(), 1);

    node = *(nodes_iter++);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r1.ref());
    UNITTEST_EQ(node->producers.size(), 0);
    UNITTEST_EQ(node->consumers.size(), 1);

    node = *(nodes_iter++);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r2.ref());
    UNITTEST_EQ(node->producers.size(), 0);
    UNITTEST_EQ(node->consumers.size(), 1);

    node = *(nodes_iter++);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t5.ref());
    UNITTEST_EQ(node->producers.size(), 3);
    UNITTEST_EQ(node->consumers.size(), 0);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_model_op_sharding);
    return 0;
}
