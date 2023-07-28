// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include "logging.h"
#include "sched_opgraph.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sched_opgraph_merge()
{
    ark::Model model;

    // Basic Test.
    // Model graph:
    //
    //   TensorOp --> t0 --+--> AddOp --> t2
    //                     |
    //   TensorOp --> t1 --+
    //                     |
    //   TensorOp --> tx --+  (tx is the output reference, hidden from the code)
    //

    ark::Tensor *t0 = model.tensor({1}, ark::FP32);
    ark::Tensor *t1 = model.tensor({1}, ark::FP32);
    ark::Tensor *t2 = model.add(t0, t1);
    UNITTEST_TRUE(model.verify());

    // MergedOp graph (parentheses indicate a MergedOp):
    //
    //   (AddOp,)
    //

    auto mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 1UL);

    auto mop = mops.front().get();
    UNITTEST_EQ(mop->ops.size(), 1UL);
    UNITTEST_EQ(mop->ops[0]->outputs[0], t2);
    UNITTEST_EQ(mop->ops[0]->inputs[0], t0);
    UNITTEST_EQ(mop->ops[0]->inputs[1], t1);
    UNITTEST_EQ(mop->users.size(), 0UL);
    UNITTEST_EQ(mop->producers.size(), 0UL);

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

    ark::Tensor *t3 = model.add(t2, t1);
    UNITTEST_TRUE(model.verify());

    // MergedOp graph (parentheses indicate a MergedOp):
    //
    //   (AddOp,AddOp,)
    //

    mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 1UL);

    mop = mops.front().get();

    UNITTEST_EQ(mop->ops[0]->outputs[0], t2);
    UNITTEST_EQ(mop->ops[0]->inputs[0], t0);
    UNITTEST_EQ(mop->ops[0]->inputs[1], t1);
    UNITTEST_EQ(mop->ops[1]->outputs[0], t3);
    UNITTEST_EQ(mop->ops[1]->inputs[0], t2);
    UNITTEST_EQ(mop->ops[1]->inputs[1], t1);
    UNITTEST_EQ(mop->users.size(), 0UL);
    UNITTEST_EQ(mop->producers.size(), 0UL);

    // Test a chain of Ops without shared input tensors.
    // Model graph (omit leftmost part):
    //
    // ... ----+--> AddOp --> t3 ----+-> ReluOp --> t4
    // ...     |                     |
    // ... ----+   TensorOp --> tz --+
    // ...     |
    // ...   --+   (tz is the output reference, hidden from the code)
    //

    ark::Tensor *t4 = model.relu(t3);
    UNITTEST_TRUE(model.verify());

    // MergedOp graph (parentheses indicate a MergedOp):
    //
    //   (AddOp,AddOp,ReluOp,)
    //

    mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 1UL);

    mop = mops.front().get();

    UNITTEST_EQ(mop->ops[0]->outputs[0], t2);
    UNITTEST_EQ(mop->ops[0]->inputs[0], t0);
    UNITTEST_EQ(mop->ops[0]->inputs[1], t1);
    UNITTEST_EQ(mop->ops[1]->outputs[0], t3);
    UNITTEST_EQ(mop->ops[1]->inputs[0], t2);
    UNITTEST_EQ(mop->ops[1]->inputs[1], t1);
    UNITTEST_EQ(mop->ops[2]->outputs[0], t4);
    UNITTEST_EQ(mop->ops[2]->inputs[0], t3);
    UNITTEST_EQ(mop->users.size(), 0UL);
    UNITTEST_EQ(mop->producers.size(), 0UL);

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

    ark::Tensor *t5 = model.add(t2, t4);
    UNITTEST_TRUE(model.verify());

    // MergedOp graph (parentheses indicate a MergedOp):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,)
    //

    mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 3UL);

    auto mops_iter = mops.begin();
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t2);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t3);
    UNITTEST_EQ(mop->ops[1]->outputs[0], t4);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t5);

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

    ark::Tensor *t6 = model.tensor({1}, ark::FP32);
    ark::Tensor *t7 = model.tensor({1}, ark::FP32);
    ark::Tensor *t8 = model.add(t6, t7);
    ark::Tensor *t9 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // MergedOp graph (parentheses indicate a MergedOp):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,) --+
    //                                                     |
    //                                          (AddOp,) --+--> (AddOp,)
    //

    mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 5UL);

    mops_iter = mops.begin();
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t2);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t3);
    UNITTEST_EQ(mop->ops[1]->outputs[0], t4);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t5);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t8);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t9);

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

    ark::Tensor *t10 = model.tensor({1}, ark::FP32);
    ark::Tensor *t11 = model.add(t10, t10);
    UNITTEST_TRUE(model.verify());

    // MergedOp graph (parentheses indicate a MergedOp):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,) --+
    //                                                     |
    //                                          (AddOp,) --+--> (AddOp,)
    //
    //                                                          (AddOp,)
    //

    mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 6UL);

    mops_iter = mops.begin();
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t2);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t3);
    UNITTEST_EQ(mop->ops[1]->outputs[0], t4);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t5);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t8);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t9);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t11);

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

    ark::Tensor *t12 = model.add(t5, t8);
    UNITTEST_TRUE(model.verify());

    // MergedOp graph (parentheses indicate a MergedOp):
    //
    //              +----------------------+
    //              |                      |
    //   (AddOp,) --+--> (AddOp,ReluOp,) --+--> (AddOp,) --+--> (AddOp,)
    //                                                     |
    //                                          (AddOp,) --+--> (AddOp,)
    //
    //                                                          (AddOp,)
    //

    mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 7UL);

    mops_iter = mops.begin();
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t2);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t3);
    UNITTEST_EQ(mop->ops[1]->outputs[0], t4);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t5);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t8);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t9);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t11);
    mop = (mops_iter++)->get();
    UNITTEST_EQ(mop->ops[0]->outputs[0], t12);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_opgraph_merge_noop()
{
    ark::Model model;
    model.tensor({1}, ark::FP32);
    model.tensor({1}, ark::FP32);
    model.tensor({1}, ark::FP32);
    UNITTEST_TRUE(model.verify());

    auto mops = ark::OpGraph::merge_ops(model);
    UNITTEST_EQ(mops.size(), 0UL);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sched_opgraph_merge);
    UNITTEST(test_sched_opgraph_merge_noop);
    return 0;
}
