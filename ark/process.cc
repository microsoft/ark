// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "ark/include/ark_utils.h"
#include "ark/logging.h"

using namespace std;

namespace ark {

// Spawn a process that runs `func`. Returns PID of the spawned process.
int proc_spawn(const function<int()> &func)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    } else if (pid == 0) {
        int ret = func();
        std::exit(ret);
    }
    return (int)pid;
}

// Wait for a spawned process with PID `pid`.
// Return -1 on any unexpected failure, otherwise return the exit status.
int proc_wait(int pid)
{
    int status;
    if (waitpid(pid, &status, 0) == -1) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

// Wait for multiple child processes.
// Return 0 on success, -1 on any unexpected failure, otherwise the first seen
// non-zero exit status.
int proc_wait(const vector<int> &pids)
{
    int ret = 0;
    for (auto &pid : pids) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            LOG(WARN, "waitpid failed (", errno, ")");
            return -1;
        }
        int r;
        if (WIFEXITED(status)) {
            r = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            LOG(WARN, "PID ", pid, " exited by signal ", WTERMSIG(status));
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

} // namespace ark
