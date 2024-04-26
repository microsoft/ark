// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ipc/ipc_hosts.h"

#include "env.h"
#include "file_io.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_ipc_hosts() {
    auto tmp_dir = ark::get_env().path_tmp_dir;
    auto tmp_hostfile = tmp_dir + "/.test_ipc_hostfile";
    ark::write_file(tmp_hostfile, "127.0.0.1\n127.0.0.1\n127.0.0.1\n");
    ::setenv("ARK_HOSTFILE", tmp_hostfile.c_str(), 1);
    ark::init();

    UNITTEST_EQ(ark::get_host(0, true), "127.0.0.1");
    UNITTEST_EQ(ark::get_host(1), "127.0.0.1");
    UNITTEST_EQ(ark::get_host(2), "127.0.0.1");

    UNITTEST_THROW(ark::get_host(-1), ark::InvalidUsageError);
    UNITTEST_THROW(ark::get_host(3), ark::InvalidUsageError);

    ark::remove_file(tmp_hostfile);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_ipc_hosts_unknown_host() {
    auto tmp_dir = ark::get_env().path_tmp_dir;
    auto tmp_hostfile = tmp_dir + "/.test_ipc_hostfile";
    ark::write_file(tmp_hostfile, "unknown\nunknown\nunknown\n");
    ::setenv("ARK_HOSTFILE", tmp_hostfile.c_str(), 1);
    ark::init();

    UNITTEST_THROW(ark::get_host(0, true), ark::InvalidUsageError);

    ark::remove_file(tmp_hostfile);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_ipc_hosts);
    UNITTEST(test_ipc_hosts_unknown_host);
    return 0;
}
