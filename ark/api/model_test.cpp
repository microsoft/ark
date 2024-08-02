// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"

#include <algorithm>

#include "logging.h"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_model_basics() {
    ark::Model model;
    ark::Model compressed;

    // Basic Test.
    // Model graph:
    //
    //   TensorOp --> t0 --+--> AddOp --> t2
    //                     |
    //   TensorOp --> t1 --+
    //                     |
    //   TensorOp --> tx --+  (tx is a write_tensor, hidden from the code)
    //

    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);
    ark::Tensor t2 = model.add(t0, t1);

    UNITTEST_TRUE(model.verify());
    UNITTEST_FALSE(model.compressed());

    // OpNode graph:
    //
    //   AddOp
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());
    UNITTEST_TRUE(compressed.compressed());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 1);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[0]->op->read_tensors()[0], t0.ref());
    UNITTEST_EQ(nodes[0]->op->read_tensors()[1], t1.ref());
    UNITTEST_EQ(nodes[0]->consumers.size(), 0);
    UNITTEST_EQ(nodes[0]->producers.size(), 0);

    // Test a chain of Ops that share a read_tensor.
    // Model graph:
    //
    // TensorOp --> t0 --+--> AddOp --> t2 ------+--> AddOp --> t3
    //                   |                       |
    // TensorOp --> t1 --+-----------------------+
    //                   |                       |
    // TensorOp --> tx --+     TensorOp --> ty --+
    //
    // (tx and ty are write_tensors, hidden from the code)
    //

    ark::Tensor t3 = model.add(t2, t1);

    UNITTEST_TRUE(model.verify());

    // OpNode graph:
    //
    //   AddOp --> AddOp
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 2);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[0]->op->read_tensors()[0], t0.ref());
    UNITTEST_EQ(nodes[0]->op->read_tensors()[1], t1.ref());
    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], t3.ref());
    UNITTEST_EQ(nodes[1]->op->read_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[1]->op->read_tensors()[1], t1.ref());

    UNITTEST_EQ(nodes[0]->consumers.size(), 1);
    UNITTEST_EQ(nodes[0]->producers.size(), 0);
    UNITTEST_EQ(nodes[1]->consumers.size(), 0);
    UNITTEST_EQ(nodes[1]->producers.size(), 1);

    // Test a chain of Ops without shared read_tensors.
    // Model graph (omit leftmost part):
    //
    // ... ----+--> AddOp --> t3 ----+-> ReluOp --> t4
    // ...     |                     |
    // ... ----+   TensorOp --> tz --+
    // ...     |
    // ...   --+   (tz is a write_tensor, hidden from the code)
    //

    ark::Tensor t4 = model.relu(t3);

    UNITTEST_TRUE(model.verify());

    // OpNode graph:
    //
    //   AddOp --> AddOp --> ReluOp
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 3);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[0]->op->read_tensors()[0], t0.ref());
    UNITTEST_EQ(nodes[0]->op->read_tensors()[1], t1.ref());
    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], t3.ref());
    UNITTEST_EQ(nodes[1]->op->read_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[1]->op->read_tensors()[1], t1.ref());
    UNITTEST_EQ(nodes[2]->op->result_tensors()[0], t4.ref());
    UNITTEST_EQ(nodes[2]->op->read_tensors()[0], t3.ref());

    UNITTEST_EQ(nodes[0]->consumers.size(), 1);
    UNITTEST_EQ(nodes[0]->producers.size(), 0);
    UNITTEST_EQ(nodes[1]->consumers.size(), 1);
    UNITTEST_EQ(nodes[1]->producers.size(), 1);
    UNITTEST_EQ(nodes[2]->consumers.size(), 0);
    UNITTEST_EQ(nodes[2]->producers.size(), 1);

    // Test a chain of Ops that use the result_tensor from the same previous Op.
    // Model graph (omit leftmost part):
    //
    // ...   +---- (this is t2) -------------------------+--> AddOp --> t5
    // ...   |                                           |
    // ... --+-+--> AddOp --> t3 ----+-> ReluOp --> t4 --+
    // ...     |                     |                   |
    // ... ----+   TensorOp --> tz --+                   |
    // ...     |                       TensorOp --> tw --+
    // ...   --+
    //
    // (tz and tw are write_tensors, hidden from the code)
    //

    ark::Tensor t5 = model.add(t2, t4);
    UNITTEST_TRUE(model.verify());

    // OpNode graph:
    //
    //   AddOp --> AddOp --> ReluOp --> AddOp
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], t3.ref());
    UNITTEST_EQ(nodes[2]->op->result_tensors()[0], t4.ref());
    UNITTEST_EQ(nodes[3]->op->result_tensors()[0], t5.ref());

    // Test an Op that uses result_tensors from multiple previous Ops.
    // Model graph (omit leftmost part):
    //
    // ... ----- (this is t2) --+--> AddOp --> t5
    // ...                      |              |
    // ... -+-> ReluOp --> t4 --+              |
    // ...  |                   |              |
    // ... -+                   |              |
    // ...    TensorOp --> tw --+              |
    // ...                                     |
    //                                         |
    //   TensorOp --> t6 --+--> AddOp --> t8 --+--> AddOp --> t9
    //                     |
    //   TensorOp --> t7 --+
    //                     |
    //   TensorOp --> tu --+
    //
    // (tw and tu are write_tensors, hidden from the code)
    //

    ark::Tensor t6 = model.tensor({1}, ark::FP32);
    ark::Tensor t7 = model.tensor({1}, ark::FP32);
    ark::Tensor t8 = model.add(t6, t7);
    ark::Tensor t9 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // OpNode graph:
    //
    //   AddOp --> AddOp --> ReluOp --> AddOp --+
    //                                          |
    //                                  AddOp --+--> AddOp
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 6);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], t3.ref());
    UNITTEST_EQ(nodes[2]->op->result_tensors()[0], t4.ref());
    UNITTEST_EQ(nodes[3]->op->result_tensors()[0], t5.ref());
    UNITTEST_EQ(nodes[4]->op->result_tensors()[0], t8.ref());
    UNITTEST_EQ(nodes[5]->op->result_tensors()[0], t9.ref());

    // Test an Op that uses a single tensor for multiple inputs.
    // Model graph (omit leftmost part):
    //
    // ... ----- (this is t2) --+--> AddOp --> t5
    // ...                      |              |
    // ... -+-> ReluOp --> t4 --+              |
    // ...  |                   |              |
    // ... -+                   |              |
    // ...    TensorOp --> tw --+              |
    // ...                                     |
    //                                         |
    //   TensorOp --> t6 --+--> AddOp --> t8 --+--> AddOp --> t9
    //                     |
    //   TensorOp --> t7 --+
    //                     |
    //   TensorOp --> tu --+
    //
    //   TensorOp --> t10 --+--> AddOp --> t11
    //                      |    ^  ^
    //                      |    |  |
    //                      +----+  |
    //                              |
    //   TensorOp --> tv -----------+
    //
    // (tw, tu, and tv are write_tensors, hidden from the code)
    //

    ark::Tensor t10 = model.tensor({1}, ark::FP32);
    ark::Tensor t11 = model.add(t10, t10);
    UNITTEST_TRUE(model.verify());

    // OpNode graph:
    //
    //   AddOp --> AddOp --> ReluOp --> AddOp --+
    //                                          |
    //                                  AddOp --+--> AddOp
    //
    //                                               AddOp
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 7);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], t3.ref());
    UNITTEST_EQ(nodes[2]->op->result_tensors()[0], t4.ref());
    UNITTEST_EQ(nodes[3]->op->result_tensors()[0], t5.ref());
    UNITTEST_EQ(nodes[4]->op->result_tensors()[0], t8.ref());
    UNITTEST_EQ(nodes[5]->op->result_tensors()[0], t9.ref());
    UNITTEST_EQ(nodes[6]->op->result_tensors()[0], t11.ref());

    // Test using previous Ops' result_tensors from multiple different Ops.
    // Model graph (omit leftmost part):
    //
    // ... ----- (this is t2) --+--> AddOp --> t5
    // ...                      |              |
    // ... -+-> ReluOp --> t4 --+              |
    // ...  |                   |              |
    // ... -+                   |              |
    // ...    TensorOp --> tw --+              |
    // ...                                     |
    //                                         |
    //   TensorOp --> t6 --+--> AddOp --> t8 --+--> AddOp --> t9
    //                     |                   |
    //   TensorOp --> t7 --+                   +--> AddOp --> t12
    //                     |
    //   TensorOp --> tu --+
    //
    //   TensorOp --> t10 --+--> AddOp --> t11
    //                      |    ^  ^
    //                      |    |  |
    //                      +----+  |
    //                              |
    //   TensorOp --> tv -----------+
    //
    // (tw, tu, and tv are write_tensors, hidden from the code)
    //

    ark::Tensor t12 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // OpNode graph:
    //
    //   AddOp --> AddOp --> ReluOp --> AddOp --+--> AddOp
    //                                          |
    //                                  AddOp --+--> AddOp
    //
    //                                               AddOp
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 8);

    UNITTEST_EQ(nodes[0]->op->result_tensors()[0], t2.ref());
    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], t3.ref());
    UNITTEST_EQ(nodes[2]->op->result_tensors()[0], t4.ref());
    UNITTEST_EQ(nodes[3]->op->result_tensors()[0], t5.ref());
    UNITTEST_EQ(nodes[4]->op->result_tensors()[0], t8.ref());
    UNITTEST_EQ(nodes[5]->op->result_tensors()[0], t9.ref());
    UNITTEST_EQ(nodes[6]->op->result_tensors()[0], t11.ref());
    UNITTEST_EQ(nodes[7]->op->result_tensors()[0], t12.ref());

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_model_dependent_inputs() {
    ark::Model m;

    ark::Tensor ones = m.tensor({256, 256}, ark::FP16);
    ark::Tensor x0 = m.mul(m.mul(ones, 2), 2);
    ark::Tensor x1 = m.mul(m.mul(x0, 2), 2);

    ark::Tensor x2 = m.mul(ones, x1);
    ark::Tensor x3 = m.mul(ones, x1);
    ark::Tensor x4 = m.mul(x2, x3);
    ark::Tensor y = m.add(x0, x4);

    // OpNode graph:
    //
    //                   x0                  x1         x2         x4
    //   MulOp -> MulOp -+-> MulOp -> MulOp -+-> MulOp -+-> MulOp -+-> AddOp
    //                   |                   |          |          |
    //                   |                   +-> MulOp -+ x3       |
    //                   +-----------------------------------------+
    //

    auto compressed = m.compress();
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 8);

    UNITTEST_EQ(nodes[1]->op->result_tensors()[0], x0.ref());
    UNITTEST_EQ(nodes[1]->consumers.size(), 2);
    UNITTEST_EQ(nodes[1]->producers.size(), 1);

    UNITTEST_EQ(nodes[3]->op->result_tensors()[0], x1.ref());
    UNITTEST_EQ(nodes[3]->consumers.size(), 2);
    UNITTEST_EQ(nodes[3]->producers.size(), 1);

    UNITTEST_EQ(nodes[4]->op->result_tensors()[0], x2.ref());
    UNITTEST_EQ(nodes[4]->consumers.size(), 1);
    UNITTEST_EQ(nodes[4]->producers.size(), 1);

    UNITTEST_EQ(nodes[5]->op->result_tensors()[0], x3.ref());
    UNITTEST_EQ(nodes[5]->consumers.size(), 1);
    UNITTEST_EQ(nodes[5]->producers.size(), 1);

    UNITTEST_EQ(nodes[6]->op->result_tensors()[0], x4.ref());
    UNITTEST_EQ(nodes[6]->consumers.size(), 1);
    UNITTEST_EQ(nodes[6]->producers.size(), 2);

    UNITTEST_EQ(nodes[7]->op->result_tensors()[0], y.ref());
    UNITTEST_EQ(nodes[7]->consumers.size(), 0);
    UNITTEST_EQ(nodes[7]->producers.size(), 2);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_model_noop() {
    ark::Model model;
    model.tensor({1}, ark::FP32);
    model.tensor({1}, ark::FP32);
    model.tensor({1}, ark::FP32);

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());
    UNITTEST_EQ(compressed.nodes().size(), 0);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_model_cumulate() {
    // OpNode graph:
    //
    //           Relu --+   Relu --+
    //                  |          |
    //   Relu --> Add --+--> Add --+--> Add
    //

    ark::Model model;
    ark::Tensor cumulate = model.tensor({1}, ark::FP32);

    for (int i = 0; i < 3; ++i) {
        ark::Tensor t = model.tensor({1}, ark::FP32);
        ark::Tensor r = model.relu(t);
        cumulate = model.add(cumulate, r);
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 6);

    auto last_node = nodes.back().get();
    UNITTEST_EQ(last_node->op->result_tensors()[0], cumulate.ref());
    UNITTEST_EQ(last_node->producers.size(), 2);
    UNITTEST_EQ(last_node->consumers.size(), 0);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_model_basics);
    UNITTEST(test_model_dependent_inputs);
    UNITTEST(test_model_noop);
    UNITTEST(test_model_cumulate);
    return 0;
}
