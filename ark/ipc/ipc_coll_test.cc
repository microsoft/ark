// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/ipc/ipc_coll.h"
#include "ark/unittest/unittest_utils.h"

using namespace ark;
using namespace std;

unittest::State test_ipc_coll_allgather()
{
    // Launch 100 workers.
    vector<int> pids;
    for (int i = 0; i < 100; ++i) {
        int pid = ark::utils::proc_spawn([i]() {
            unittest::Timeout timeout{5};
            int rank = i;
            int size = 100;
            int data = rank + 1;
            IpcAllGather iag{"ipc_coll_allgather_test", rank, size,
                             (const void *)&data, sizeof(data)};
            iag.sync();
            for (int i = 0; i < size; ++i) {
                int *ptr = (int *)iag.get_data(i);
                UNITTEST_EQ(*ptr, i + 1);
            }
            return 0;
        });
        UNITTEST_NE(pid, -1);
        pids.emplace_back(pid);
    }
    // Wait until all workers finish.
    int ret = ark::utils::proc_wait(pids);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_ipc_coll_allgather);
    return 0;
}
