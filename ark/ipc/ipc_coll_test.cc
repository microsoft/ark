// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ipc/ipc_coll.h"

#include "include/ark.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_ipc_coll_allgather() {
    // Launch 100 workers.
    for (int i = 0; i < 100; ++i) {
        int pid = ark::unittest::spawn_process([i]() {
            ark::unittest::Timeout timeout{5};
            int rank = i;
            int size = 100;
            int data = rank + 1;
            ark::IpcAllGather iag{"ipc_coll_allgather_test", rank, size,
                                  (const void *)&data, sizeof(data)};
            iag.sync();
            for (int i = 0; i < size; ++i) {
                int *ptr = (int *)iag.get_data(i);
                UNITTEST_EQ(*ptr, i + 1);
            }
            return ark::unittest::SUCCESS;
        });
        UNITTEST_NE(pid, -1);
    }
    // Wait until all workers finish.
    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_ipc_coll_allgather);
    return 0;
}
