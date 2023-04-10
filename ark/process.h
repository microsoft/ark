#ifndef ARK_PROCESS_H_
#define ARK_PROCESS_H_

#include <functional>
#include <vector>

namespace ark {

// Spawn a process that runs `func`. Returns PID of the spawned process.
int proc_spawn(const std::function<int()> &func);
// Wait for a spawned process with PID `pid`.
// Return -1 on any unexpected failures, otherwise return the exit status.
int proc_wait(int pid);
// Wait for multiple child processes.
// Return 0 on success, -1 on any unexpected failure, otherwise the first seen
// non-zero exit status.
int proc_wait(const std::vector<int> &pids);

} // namespace ark

#endif // ARK_PROCESS_H_
