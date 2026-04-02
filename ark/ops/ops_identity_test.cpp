// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "ops_test_common.hpp"

ark::unittest::State test_ops_identity_model() {
    // OpNode graph:
    //
    //   ReluOp --+
    //            |
    //   ReluOp --+--> ReluOp
    //

    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);
    ark::Tensor t2 = model.tensor({1}, ark::FP32);

    ark::Tensor r0 = model.relu(t0);
    ark::Tensor r1 = model.relu(t1);
    ark::Tensor t3 = model.identity(t2, {r0, r1});

    ark::Tensor t4 = model.relu(t3);
    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 3);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], r0.ref());
    UNITTEST_EQ(nodes[0]->producers.size(), 0);
    UNITTEST_EQ(nodes[0]->consumers.size(), 1);

    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], r1.ref());
    UNITTEST_EQ(nodes[1]->producers.size(), 0);
    UNITTEST_EQ(nodes[1]->consumers.size(), 1);

    UNITTEST_EQ(nodes[2]->op->result_tensors()[0], t4.ref());
    UNITTEST_EQ(nodes[2]->producers.size(), 2);
    UNITTEST_EQ(nodes[2]->consumers.size(), 0);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_ops_identity() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor tns1 = model.identity(tns0);

    // For preventing optimize-out
    model.noop(tns0);
    model.noop(tns1);

    // Create an executor
    ark::DefaultExecutor exe(model);

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
    UNITTEST(test_ops_identity_model);
    UNITTEST(test_ops_identity);
    return 0;
}
