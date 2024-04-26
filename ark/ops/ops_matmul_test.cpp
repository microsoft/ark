// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>

#include "ark/model.hpp"
#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_model_op_matmul() {
    // Hidden dimension of the dense layer.
    unsigned int units = 1024;
    // Input dimension of the dense layer.
    unsigned int in_dim = 1024;
    // Extra dimension of the input. CHANNEL=1 for 2D inputs.
    unsigned int channel = 128;
    // Batch size of the input.
    unsigned int batch_size = 1;

    ark::Model m;
    ark::ModelTensorRef input =
        m.tensor({batch_size, channel, in_dim}, ark::FP16);
    ark::ModelTensorRef weight = m.tensor({in_dim, units}, ark::FP16);
    m.matmul(input, weight);

    UNITTEST_TRUE(m.verify());
    auto compressed = m.compress();
    UNITTEST_TRUE(compressed.verify());

    return ark::unittest::SUCCESS;
}

// ark::unittest::State test_model_op_split_matmul() {
//     // OpNode graph (parentheses indicate a OpNode):
//     //
//     //   (Matmul,) --+
//     //               |
//     //   (Matmul,) --+--> (Reduce,)
//     //

//     ark::Model model;
//     ark::ModelTensorRef t0 = model.tensor({64, 128}, ark::FP16);
//     ark::ModelTensorRef t1 = model.tensor({128, 64}, ark::FP16);
//     model.matmul(t0, t1, nullptr, 2, false, false, "matmul", 3);
//     UNITTEST_TRUE(model.verify());

//     auto compressed = model.compress();
//     auto nodes = compressed.nodes();
//     UNITTEST_EQ(nodes.size(), 3);

//     auto nodes_iter = nodes.begin();
//     auto node = (nodes_iter++)->get();
//     // UNITTEST_EQ(node->ops[0]->name, "matmul/matmul_shard_0");
//     UNITTEST_EQ(node->producers.size(), 0);
//     UNITTEST_EQ(node->consumers.size(), 1);

//     node = (nodes_iter++)->get();
//     // UNITTEST_EQ(node->ops[0]->name, "matmul/matmul_shard_1");
//     UNITTEST_EQ(node->producers.size(), 0);
//     UNITTEST_EQ(node->consumers.size(), 1);

//     node = (nodes_iter++)->get();
//     // UNITTEST_EQ(node->ops[0]->name, "matmul/reduce_sum");
//     UNITTEST_EQ(node->producers.size(), 2);
//     UNITTEST_EQ(node->consumers.size(), 0);

//     return ark::unittest::SUCCESS;
// }

int main() {
    UNITTEST(test_model_op_matmul);
    // UNITTEST(test_model_op_split_matmul);
    return 0;
}
