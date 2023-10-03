// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Example usage of KaHyPar interface in ARK.
// Environment variable `ARK_ROOT` should be set to run.
// `LD_LIBRARY_PATH` should include `$ARK_ROOT/lib` directory.

#include "kahypar.h"

#include "include/ark.h"
#include "unittest/unittest_utils.h"

#define ITERATION 3

using namespace std;

struct TestObj {
    TestObj(int id_) : id(id_) {}
    int id;

    bool operator<(const TestObj &rhs) const { return id < rhs.id; }
};

function<TestObj *()> gen(initializer_list<int> seq) {
    return [&] {
        const vector<int> seq_(seq);
        size_t cnt = 0;
        return [=]() mutable {
            return cnt < seq.size() ? new TestObj(seq_[cnt++]) : nullptr;
        };
    }();
}

ark::unittest::State test_simple() {
    ark::KahyparGraph<TestObj> kg;
    kg.add_nodes(1, gen({0, 1, 2, 3, 4, 5, 6}));

    kg.add_edge(1, gen({0, 2}));
    kg.add_edge(1000, gen({0, 1, 3, 4}));
    kg.add_edge(1, gen({3, 4, 6}));
    kg.add_edge(1000, gen({2, 5, 6}));

    auto &parts = kg.partition(2);

    // Print results. Desired outputs:
    // res[0] = { 2 5 6 } and res[1] = { 3 4 0 1 }, or
    // res[0] = { 3 4 0 1 } and res[1] = { 2 5 6 }.
    vector<TestObj *> res[2];
    for (int idx = 0; idx < 2; ++idx) {
        for (auto &p : parts[idx]) {
            res[idx].push_back(p.first);
            if (p.second) res[idx].push_back(p.second);
        }
    }

    // Verify results.
    if (res[0][0]->id == 2) {
        UNITTEST_EQ(res[0][0]->id, 2);
        UNITTEST_EQ(res[0][1]->id, 5);
        UNITTEST_EQ(res[0][2]->id, 6);
        UNITTEST_EQ(res[1][0]->id, 3);
        UNITTEST_EQ(res[1][1]->id, 4);
        UNITTEST_EQ(res[1][2]->id, 0);
        UNITTEST_EQ(res[1][3]->id, 1);
    } else {
        UNITTEST_EQ(res[0][0]->id, 3);
        UNITTEST_EQ(res[0][1]->id, 4);
        UNITTEST_EQ(res[0][2]->id, 0);
        UNITTEST_EQ(res[0][3]->id, 1);
        UNITTEST_EQ(res[1][0]->id, 2);
        UNITTEST_EQ(res[1][1]->id, 5);
        UNITTEST_EQ(res[1][2]->id, 6);
    }

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    for (int i = 0; i < ITERATION; ++i) {
        UNITTEST(test_simple);
    }
    return 0;
}
