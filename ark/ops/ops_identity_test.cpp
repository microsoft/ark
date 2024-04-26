// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"
#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_model_op_identity() {
    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (Relu,) --+
    //             |
    //   (Relu,) --+--> (Relu,)
    //

    ark::Model model;
    ark::ModelTensorRef t0 = model.tensor({1}, ark::FP32);
    ark::ModelTensorRef t1 = model.tensor({1}, ark::FP32);
    ark::ModelTensorRef t2 = model.tensor({1}, ark::FP32);

    ark::ModelTensorRef r0 = model.relu(t0);
    ark::ModelTensorRef r1 = model.relu(t1);
    ark::ModelTensorRef t3 = model.identity(t2, {r0, r1});

    ark::ModelTensorRef t4 = model.relu(t3);
    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 3);

    auto nodes_iter = nodes.begin();
    auto node = *(nodes_iter++);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r0);
    UNITTEST_EQ(node->producers.size(), 0);
    UNITTEST_EQ(node->consumers.size(), 1);

    node = *(nodes_iter++);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r1);
    UNITTEST_EQ(node->producers.size(), 0);
    UNITTEST_EQ(node->consumers.size(), 1);

    node = *(nodes_iter++);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t4);
    UNITTEST_EQ(node->producers.size(), 2);
    UNITTEST_EQ(node->consumers.size(), 0);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_model_op_identity);
    return 0;
}
