// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <vector>

#include "include/ark.h"
#include "include/ark_utils.h"

using namespace std;

namespace ark {
namespace utils {

/// Spawn a process that runs `func`.
/// @param func function to run in the spawned process.
/// @return PID of the spawned process.
int proc_spawn(const function<int()> &func) {
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    } else if (pid == 0) {
        int ret = func();
        std::exit(ret);
    }
    return (int)pid;
}

/// Wait for a spawned process with PID `pid`.
/// @param pid PID of the spawned process.
/// @return -1 on any unexpected failure, otherwise return the exit status.
int proc_wait(int pid) {
    int status;
    if (waitpid(pid, &status, 0) == -1) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

/// Wait for multiple child processes.
/// @param pids PIDs of the spawned processes.
/// @return 0 on success, -1 on any unexpected failure, otherwise the first seen
/// non-zero exit status.
int proc_wait(const vector<int> &pids) {
    int ret = 0;
    for (auto &pid : pids) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            return -1;
        }
        int r;
        if (WIFEXITED(status)) {
            r = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            r = -1;
        } else {
            r = -1;
        }
        if ((ret == 0) && (r != 0)) {
            ret = r;
        }
    }
    return ret;
}

}  // namespace utils
}  // namespace ark
