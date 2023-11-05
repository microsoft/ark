// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ipc/ipc_mem.h"

#include "include/ark.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_ipc_mem_lock_simple() {
    auto fc = [] {
        ark::unittest::Timeout timeout{3};
        ark::IpcMem im{"ipc_mem_lock_test", true};
        volatile int *ptr = (volatile int *)im.alloc(4);
        while (*ptr == 0) {
        }
        while (*ptr == 1) {
        }
        UNITTEST_EQ(*ptr, 2);
        return ark::unittest::SUCCESS;
    };
    auto fn = [] {
        ark::unittest::Timeout timeout{3};
        ark::IpcMem im{"ipc_mem_lock_test", false};
        volatile int *ptr = (volatile int *)im.alloc(4);
        ark::IpcLockGuard lg{im.get_lock()};
        *ptr += 1;
        return ark::unittest::SUCCESS;
    };

    for (int i = 0; i < 10; ++i) {
        int idx = i % 3;
        int pid0 = ark::unittest::spawn_process(idx == 0 ? fc : fn);
        UNITTEST_NE(pid0, -1);
        int pid1 = ark::unittest::spawn_process(idx == 1 ? fc : fn);
        UNITTEST_NE(pid1, -1);
        int pid2 = ark::unittest::spawn_process(idx == 2 ? fc : fn);
        UNITTEST_NE(pid2, -1);
        ark::unittest::wait_all_processes();
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_ipc_mem_lock_many() {
    auto worker = [] {
        ark::unittest::Timeout timeout{3};
        // Elect the earliest starting worker as the creator.
        ark::IpcMem im{"ipc_mem_lock_test_many", false, true};
        volatile int *ptr = (volatile int *)im.alloc(8);
        volatile int *data = &ptr[0];
        volatile int *counter = &ptr[1];
        // Each worker increases the shared data by 10000.
        for (int i = 0; i < 10000; ++i) {
            ark::IpcLockGuard lg{im.get_lock()};
            *data += 1;
        }
        {
            // Count finished workers.
            ark::IpcLockGuard lg{im.get_lock()};
            *counter += 1;
        }
        if (im.is_create()) {
            // Wait until all other workers finish.
            while (*counter != 100) {
            }
            // Validate the result.
            UNITTEST_EQ(*data, 1000000);
        }
        return ark::unittest::SUCCESS;
    };

    // Launch 100 workers.
    for (int i = 0; i < 100; ++i) {
        int pid = ark::unittest::spawn_process(worker);
        UNITTEST_NE(pid, -1);
    }
    // Wait until all workers finish.
    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_ipc_mem_finishing() {
    int pid0 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{3};
        ark::IpcMem im{"ipc_mem_finishing", true};
        volatile int *ptr = (volatile int *)im.alloc(4);
        ptr[0] = 7;
        // Wait until another process reads the results.
        while (ptr[0] == 7) {
        }
        // Modify the data.
        ptr[0] = 77;
        // Just return without waiting for another process.
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{3};
        ark::IpcMem im{"ipc_mem_finishing", false};
        volatile int *ptr = (volatile int *)im.alloc(4);
        while (ptr[0] != 7) {
        }
        // Notification.
        ptr[0] = -1;
        // Wait for a while until the `f0` process completes.
        ark::cpu_timer_sleep(0.1);
        // Read the modified data.
        // This should work even though `f0` is already returned.
        while (ptr[0] != 77) {
        }
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid1, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_ipc_mem_realloc() {
    int pid0 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{3};
        ark::IpcMem im{"ipc_mem_realloc", true};
        volatile int *ptr = (volatile int *)im.alloc(4);
        ptr[0] = 7;
        ptr = (volatile int *)im.alloc(8);
        while (ptr[0] == 7) {
        }
        ptr[0] = 77;
        ptr[1] = 88;
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{3};
        ark::IpcMem im{"ipc_mem_realloc", false};
        volatile int *ptr = (volatile int *)im.alloc(4);
        while (ptr[0] != 7) {
        }
        ptr = (volatile int *)im.alloc(8);
        ptr[0] = -1;
        ark::cpu_timer_sleep(0.1);
        while (ptr[0] != 77) {
        }
        while (ptr[1] != 88) {
        }
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid1, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_ipc_mem_lock_simple);
    UNITTEST(test_ipc_mem_lock_many);
    UNITTEST(test_ipc_mem_finishing);
    UNITTEST(test_ipc_mem_realloc);
    return 0;
}
