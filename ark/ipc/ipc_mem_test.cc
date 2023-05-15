// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/include/ark.h"
#include "ark/ipc/ipc_mem.h"
#include "ark/process.h"
#include "ark/unittest/unittest_utils.h"

using namespace ark;
using namespace std;

unittest::State test_ipc_mem_lock_simple()
{
    function<int()> fc = [] {
        unittest::Timeout timeout{3};
        IpcMem im{"ipc_mem_lock_test", true};
        volatile int *ptr = (volatile int *)im.alloc(4);
        while (*ptr == 0) {
        }
        while (*ptr == 1) {
        }
        UNITTEST_EQ(*ptr, 2);
        return 0;
    };
    function<int()> fn = [] {
        unittest::Timeout timeout{3};
        IpcMem im{"ipc_mem_lock_test", false};
        volatile int *ptr = (volatile int *)im.alloc(4);
        IpcLockGuard lg{im.get_lock()};
        *ptr += 1;
        return 0;
    };

    for (int i = 0; i < 10; ++i) {
        int idx = i % 3;
        int pid0 = proc_spawn(idx == 0 ? fc : fn);
        UNITTEST_NE(pid0, -1);
        int pid1 = proc_spawn(idx == 1 ? fc : fn);
        UNITTEST_NE(pid1, -1);
        int pid2 = proc_spawn(idx == 2 ? fc : fn);
        UNITTEST_NE(pid2, -1);
        int ret = proc_wait({pid0, pid1, pid2});
        UNITTEST_EQ(ret, 0);
    }
    return unittest::SUCCESS;
}

unittest::State test_ipc_mem_lock_many()
{
    function<int()> worker = [] {
        unittest::Timeout timeout{3};
        // Elect the earliest starting worker as the creator.
        IpcMem im{"ipc_mem_lock_test_many", false, true};
        volatile int *ptr = (volatile int *)im.alloc(8);
        volatile int *data = &ptr[0];
        volatile int *counter = &ptr[1];
        // Each worker increases the shared data by 10000.
        for (int i = 0; i < 10000; ++i) {
            IpcLockGuard lg{im.get_lock()};
            *data += 1;
        }
        {
            // Count finished workers.
            IpcLockGuard lg{im.get_lock()};
            *counter += 1;
        }
        if (im.is_create()) {
            // Wait until all other workers finish.
            while (*counter != 100) {
            }
            // Validate the result.
            UNITTEST_EQ(*data, 1000000);
        }
        return 0;
    };

    // Launch 100 workers.
    vector<int> pids;
    for (int i = 0; i < 100; ++i) {
        int pid = proc_spawn(worker);
        UNITTEST_NE(pid, -1);
        pids.emplace_back(pid);
    }
    // Wait until all workers finish.
    int ret = proc_wait(pids);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

unittest::State test_ipc_mem_finishing()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{3};
        IpcMem im{"ipc_mem_finishing", true};
        volatile int *ptr = (volatile int *)im.alloc(4);
        ptr[0] = 7;
        // Wait until another process reads the results.
        while (ptr[0] == 7) {
        }
        // Modify the data.
        ptr[0] = 77;
        // Just return without waiting for another process.
        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{3};
        IpcMem im{"ipc_mem_finishing", false};
        volatile int *ptr = (volatile int *)im.alloc(4);
        while (ptr[0] != 7) {
        }
        // Notification.
        ptr[0] = -1;
        // Wait for a while until the `f0` process completes.
        cpu_timer_sleep(0.1);
        // Read the modified data.
        // This should work even though `f0` is already returned.
        while (ptr[0] != 77) {
        }
        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

unittest::State test_ipc_mem_realloc()
{
    int pid0 = proc_spawn([] {
        unittest::Timeout timeout{3};
        IpcMem im{"ipc_mem_realloc", true};
        volatile int *ptr = (volatile int *)im.alloc(4);
        ptr[0] = 7;
        ptr = (volatile int *)im.alloc(8);
        while (ptr[0] == 7) {
        }
        ptr[0] = 77;
        ptr[1] = 88;
        return 0;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = proc_spawn([] {
        unittest::Timeout timeout{3};
        IpcMem im{"ipc_mem_realloc", false};
        volatile int *ptr = (volatile int *)im.alloc(4);
        while (ptr[0] != 7) {
        }
        ptr = (volatile int *)im.alloc(8);
        ptr[0] = -1;
        cpu_timer_sleep(0.1);
        while (ptr[0] != 77) {
        }
        while (ptr[1] != 88) {
        }
        return 0;
    });
    UNITTEST_NE(pid1, -1);

    int ret = proc_wait({pid0, pid1});
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_ipc_mem_lock_simple);
    UNITTEST(test_ipc_mem_lock_many);
    UNITTEST(test_ipc_mem_finishing);
    UNITTEST(test_ipc_mem_realloc);
    return 0;
}
