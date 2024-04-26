// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ipc/ipc_socket.h"

#include "env.h"
#include "ipc/ipc_hosts.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

struct TestIpcSocketItem {
    int a;
    int b;
    int c;
};

ark::unittest::State test_ipc_socket_simple() {
    int pid0 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{5};
        int port_base = ark::get_env().ipc_listen_port_base;
        ark::IpcSocket is{ark::get_host(0), port_base};

        struct TestIpcSocketItem item;
        item.a = 1;
        item.b = 2;
        item.c = 3;
        is.add_item("test_item_name", &item, sizeof(item));

        struct TestIpcSocketItem remote_item;
        ark::IpcSocket::State s =
            is.query_item(ark::get_host(0), port_base + 1, "test_item_name",
                          &remote_item, sizeof(remote_item), true);
        UNITTEST_TRUE(s == ark::IpcSocket::SUCCESS);
        UNITTEST_EQ(remote_item.a, 4);
        UNITTEST_EQ(remote_item.b, 5);
        UNITTEST_EQ(remote_item.c, 6);

        const ark::IpcSocket::Item *item_ptr = is.get_item("test_item_name");
        UNITTEST_TRUE(item_ptr != nullptr);
        while (item_ptr->cnt == 0) {
            sched_yield();
        }
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{5};
        int port_base = ark::get_env().ipc_listen_port_base;
        ark::IpcSocket is{ark::get_host(0), port_base + 1};

        struct TestIpcSocketItem item;
        item.a = 4;
        item.b = 5;
        item.c = 6;
        is.add_item("test_item_name", &item, sizeof(item));

        struct TestIpcSocketItem remote_item;
        ark::IpcSocket::State s =
            is.query_item(ark::get_host(0), port_base, "test_item_name",
                          &remote_item, sizeof(remote_item), true);
        UNITTEST_TRUE(s == ark::IpcSocket::SUCCESS);
        UNITTEST_EQ(remote_item.a, 1);
        UNITTEST_EQ(remote_item.b, 2);
        UNITTEST_EQ(remote_item.c, 3);

        const ark::IpcSocket::Item *item_ptr = is.get_item("test_item_name");
        UNITTEST_TRUE(item_ptr != nullptr);
        while (item_ptr->cnt == 0) {
            sched_yield();
        }
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid1, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_ipc_socket_no_item() {
    int pid0 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        int port_base = ark::get_env().ipc_listen_port_base;
        ark::IpcSocket is{ark::get_host(0), port_base};

        struct TestIpcSocketItem remote_item;
        ark::IpcSocket::State s =
            is.query_item(ark::get_host(0), port_base + 1, "test_item_name",
                          &remote_item, sizeof(remote_item), true);
        UNITTEST_TRUE(s == ark::IpcSocket::SUCCESS);
        UNITTEST_EQ(remote_item.a, 4);
        UNITTEST_EQ(remote_item.b, 5);
        UNITTEST_EQ(remote_item.c, 6);
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid0, -1);

    int pid1 = ark::unittest::spawn_process([] {
        ark::unittest::Timeout timeout{10};
        int port_base = ark::get_env().ipc_listen_port_base;
        ark::IpcSocket is{ark::get_host(0), port_base + 1};

        // Sleep for a while to make the remote experience NO_ITEM
        ark::cpu_timer_sleep(2);

        struct TestIpcSocketItem item;
        item.a = 4;
        item.b = 5;
        item.c = 6;
        is.add_item("test_item_name", &item, sizeof(item));

        const ark::IpcSocket::Item *item_ptr = is.get_item("test_item_name");
        UNITTEST_TRUE(item_ptr != nullptr);
        while (item_ptr->cnt == 0) {
            sched_yield();
        }
        return ark::unittest::SUCCESS;
    });
    UNITTEST_NE(pid1, -1);

    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_ipc_socket_simple);
    UNITTEST(test_ipc_socket_no_item);
    return ark::unittest::SUCCESS;
}
