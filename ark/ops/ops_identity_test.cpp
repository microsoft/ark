// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"
#include "ark/model.hpp"
#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_identity_model() {
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

ark::unittest::State test_identity() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::ModelTensorRef tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::ModelTensorRef tns1 = model.identity(tns0);

    // For preventing optimize-out
    model.noop(tns0);
    model.noop(tns1);

    // Create an executor
    ark::DefaultExecutor exe(model);
    exe.compile();

    int num_elem = 2 * 3 * 4 * 5;

    // Fill tensor data: {1.0, 2.0, 3.0, ..., 120.0}
    std::vector<float> data_vec(num_elem);
    std::iota(data_vec.begin(), data_vec.end(), 1.0f);
    exe.tensor_write(tns0, data_vec);

    // Check identity values
    std::vector<float> ref_val(num_elem);
    exe.tensor_read(tns1, ref_val);
    for (int i = 0; i < num_elem; ++i) {
        UNITTEST_EQ(ref_val[i], (float)(i + 1));
    }

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_identity_model);
    UNITTEST(test_identity);
    return 0;
}
