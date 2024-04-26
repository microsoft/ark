// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"
#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_model_op_all_reduce() {
    // OpNode graph (parentheses indicate a OpNode):
    //
    //               +--> (S,SD,R,) --+--> (S,SD,R,) --+
    //               |                |                |
    //   (S,SD,R,) --+--> (Add,)      +--> (Add,)      +--> (Add,)
    //                      |               ^  |              ^
    //                      |               |  |              |
    //                      +---------------+  +--------------+

    ark::Model model;
    ark::ModelTensorRef input = model.tensor({1}, ark::FP32);
    ark::ModelTensorRef output = model.all_reduce(input, 0, 4);

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 6);

    auto nodes_iter = nodes.begin();
    auto node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "send;send_done;recv;");
    UNITTEST_EQ(node->producers.size(), 0);
    UNITTEST_EQ(node->consumers.size(), 2);

    // UNITTEST_EQ(node->consumers[0]->get_name(), "add;");
    UNITTEST_EQ(node->consumers[0]->consumers.size(), 1);
    // UNITTEST_EQ((*(node->consumers[0]->consumers.begin()))->get_name(),
    // "add_1;");

    // UNITTEST_EQ(node->consumers[1]->get_name(),
    // "send_1;send_done_1;recv_1;");
    UNITTEST_EQ(node->consumers[1]->producers.size(), 1);
    UNITTEST_EQ(node->consumers[1]->consumers.size(), 2);

    node = node->consumers[1];

    // UNITTEST_EQ(node->consumers[0]->get_name(), "add_1;");
    UNITTEST_EQ(node->consumers[0]->producers.size(), 2);
    UNITTEST_EQ(node->consumers[0]->consumers.size(), 1);
    // UNITTEST_EQ((*(node->consumers[0]->consumers.begin()))->get_name(),
    // "add_2;");

    // UNITTEST_EQ(node->consumers[1]->get_name(),
    // "send_2;send_done_2;recv_2;");
    UNITTEST_EQ(node->consumers[1]->producers.size(), 1);
    UNITTEST_EQ(node->consumers[1]->consumers.size(), 1);
    // UNITTEST_EQ((*(node->consumers[1]->consumers.begin()))->get_name(),
    // "add_2;");
    UNITTEST_EQ(
        (*(node->consumers[1]->consumers.begin()))->ops[0]->result_tensors()[0],
        output);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_model_op_all_reduce);
    return 0;
}
