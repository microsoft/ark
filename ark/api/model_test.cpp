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
    //   TensorOp --> tx --+  (tx is the output reference, hidden from the code)
    //

    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);
    ark::Tensor t2 = model.add(t0, t1);

    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,)
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());
    UNITTEST_EQ(compressed.nodes().size(), 1);

    auto node = compressed.nodes().front();
    UNITTEST_EQ(node->ops.size(), 1);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[0], t0.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[1], t1.ref());
    UNITTEST_EQ(node->consumers.size(), 0);
    UNITTEST_EQ(node->producers.size(), 0);

    // Test a chain of Ops that share an input tensor.
    // Model graph:
    //
    // TensorOp --> t0 --+--> AddOp --> t2 ------+--> AddOp --> t3
    //                   |                       |
    // TensorOp --> t1 --+-----------------------+
    //                   |                       |
    // TensorOp --> tx --+     TensorOp --> ty --+
    //
    // (tx and ty are output references, hidden from the code)
    //

    ark::Tensor t3 = model.add(t2, t1);

    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,)
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());
    UNITTEST_EQ(compressed.nodes().size(), 1);

    node = compressed.nodes().front();

    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[0], t0.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[1], t1.ref());
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], t3.ref());
    UNITTEST_EQ(node->ops[1]->read_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[1]->read_tensors()[1], t1.ref());
    UNITTEST_EQ(node->consumers.size(), 0);
    UNITTEST_EQ(node->producers.size(), 0);

    // Test a chain of Ops without shared input tensors.
    // Model graph (omit leftmost part):
    //
    // ... ----+--> AddOp --> t3 ----+-> ReluOp --> t4
    // ...     |                     |
    // ... ----+   TensorOp --> tz --+
    // ...     |
    // ...   --+   (tz is the output reference, hidden from the code)
    //

    ark::Tensor t4 = model.relu(t3);

    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,ReluOp,)
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());
    UNITTEST_EQ(compressed.nodes().size(), 1);

    node = compressed.nodes().front();

    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[0], t0.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[1], t1.ref());
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], t3.ref());
    UNITTEST_EQ(node->ops[1]->read_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[1]->read_tensors()[1], t1.ref());
    UNITTEST_EQ(node->ops[2]->result_tensors()[0], t4.ref());
    UNITTEST_EQ(node->ops[2]->read_tensors()[0], t3.ref());
    UNITTEST_EQ(node->consumers.size(), 0);
    UNITTEST_EQ(node->producers.size(), 0);

    // Test a chain of Ops that use the output from the same previous Op.
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
    // (tz and tw are output references, hidden from the code)
    //

    ark::Tensor t5 = model.add(t2, t4);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,ReluOp,AddOp,)
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 1);

    auto nodes_iter = nodes.begin();
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add;add_1;relu;add_2;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], t3.ref());
    UNITTEST_EQ(node->ops[2]->result_tensors()[0], t4.ref());
    UNITTEST_EQ(node->ops[3]->result_tensors()[0], t5.ref());

    // Test an Op that uses outputs from multiple previous Ops.
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
    // (tw and tu are output references, hidden from the code)
    //

    ark::Tensor t6 = model.tensor({1}, ark::FP32);
    ark::Tensor t7 = model.tensor({1}, ark::FP32);
    ark::Tensor t8 = model.add(t6, t7);
    ark::Tensor t9 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,ReluOp,AddOp,) --+
    //                                 |
    //                      (AddOp,) --+--> (AddOp,)
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 3);

    nodes_iter = nodes.begin();
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add;add_1;relu;add_2;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], t3.ref());
    UNITTEST_EQ(node->ops[2]->result_tensors()[0], t4.ref());
    UNITTEST_EQ(node->ops[3]->result_tensors()[0], t5.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_3;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t8.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_4;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t9.ref());

    // Test an Op that uses a single input tensor for multiple inputs.
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
    // (tw, tu, and tv are output references, hidden from the code)
    //

    ark::Tensor t10 = model.tensor({1}, ark::FP32);
    ark::Tensor t11 = model.add(t10, t10);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,ReluOp,AddOp,) --+
    //                                 |
    //                      (AddOp,) --+--> (AddOp,)
    //
    //                                      (AddOp,)
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    nodes_iter = nodes.begin();
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add;add_1;relu;add_2;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], t3.ref());
    UNITTEST_EQ(node->ops[2]->result_tensors()[0], t4.ref());
    UNITTEST_EQ(node->ops[3]->result_tensors()[0], t5.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_3;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t8.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_4;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t9.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_5;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t11.ref());

    // Test using previous Ops' outputs from multiple different Ops.
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
    // (tw, tu, and tv are output references, hidden from the code)
    //

    ark::Tensor t12 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (AddOp,AddOp,ReluOp,AddOp,) --+--> (AddOp,)
    //                                 |
    //                      (AddOp,) --+--> (AddOp,)
    //
    //                                      (AddOp,)
    //

    compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 5);

    nodes_iter = nodes.begin();
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add;add_1;relu;add_2;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t2.ref());
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], t3.ref());
    UNITTEST_EQ(node->ops[2]->result_tensors()[0], t4.ref());
    UNITTEST_EQ(node->ops[3]->result_tensors()[0], t5.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_3;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t8.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_4;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t9.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_5;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t11.ref());
    node = *(nodes_iter++);
    // UNITTEST_EQ(node->get_name(), "add_6;");
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t12.ref());

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

    auto compressed = m.compress();
    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);
    auto nodes_iter = nodes.begin();
    auto node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops.size(), 4);
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], x0.ref());
    UNITTEST_EQ(node->ops[3]->result_tensors()[0], x1.ref());
    UNITTEST_EQ(node->consumers.size(), 3);
    UNITTEST_EQ(node->producers.size(), 0);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops.size(), 1);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], x2.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[0], ones.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[1], x1.ref());
    UNITTEST_EQ(node->consumers.size(), 1);
    UNITTEST_EQ(node->producers.size(), 1);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops.size(), 1);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], x3.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[0], ones.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[1], x1.ref());
    UNITTEST_EQ(node->consumers.size(), 1);
    UNITTEST_EQ(node->producers.size(), 1);
    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops.size(), 2);
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], x4.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[0], x2.ref());
    UNITTEST_EQ(node->ops[0]->read_tensors()[1], x3.ref());
    UNITTEST_EQ(node->ops[1]->result_tensors()[0], y.ref());
    UNITTEST_EQ(node->ops[1]->read_tensors()[0], x0.ref());
    UNITTEST_EQ(node->ops[1]->read_tensors()[1], x4.ref());
    UNITTEST_EQ(node->consumers.size(), 0);
    UNITTEST_EQ(node->producers.size(), 3);

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

ark::unittest::State test_model_identity() {
    // OpNode graph (parentheses indicate a OpNode):
    //
    //   (Relu,) --+
    //             |
    //   (Relu,) --+--> (Relu,)
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
    auto nodes_iter = nodes.begin();

    auto node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r0.ref());
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->consumers.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r1.ref());
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->consumers.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t4.ref());
    UNITTEST_EQ(node->producers.size(), 2UL);
    UNITTEST_EQ(node->consumers.size(), 0UL);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_model_sharding() {
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
    UNITTEST_EQ(vec.size(), 3UL);

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
    auto nodes_iter = nodes.begin();

    auto node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r0.ref());
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->consumers.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r1.ref());
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->consumers.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], r2.ref());
    UNITTEST_EQ(node->producers.size(), 0UL);
    UNITTEST_EQ(node->consumers.size(), 1UL);

    node = (nodes_iter++)->get();
    UNITTEST_EQ(node->ops[0]->result_tensors()[0], t5.ref());
    UNITTEST_EQ(node->producers.size(), 3UL);
    UNITTEST_EQ(node->consumers.size(), 0UL);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_model_cumulate() {
    // OpNode graph (parentheses indicate a OpNode):
    //
    //       (Relu,) --+   (Relu,) --+
    //                 |             |
    //   (Relu,Add,) --+--> (Add,) --+--> (Add,)
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
    UNITTEST_EQ(nodes.size(), 5);

    auto last_node = nodes.back().get();
    UNITTEST_EQ(last_node->ops[0]->result_tensors()[0], cumulate.ref());
    UNITTEST_EQ(last_node->producers.size(), 2);
    UNITTEST_EQ(last_node->consumers.size(), 0);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_model_basics);
    UNITTEST(test_model_dependent_inputs);
    UNITTEST(test_model_noop);
    UNITTEST(test_model_identity);
    UNITTEST(test_model_sharding);
    UNITTEST(test_model_cumulate);
    return 0;
}
