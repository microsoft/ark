#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "ark/logging.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

// Grep SIGALRM and exit.
static void sigalrm_timeout_handler(int sig)
{
    signal(SIGALRM, SIG_IGN);
    UNITTEST_FEXIT("timeout");
}

namespace ark {
namespace unittest {

// Temporal unittest states.
struct TempStates
{
    vector<int> pids;
    vector<thread *> threads;
};

TempStates GLOBAL_TEMP_STATES_;

// Set a timeout of the current process.
Timeout::Timeout(int timeout)
{
    signal(SIGALRM, sigalrm_timeout_handler);
    alarm(timeout);
}

// Remove the timeout.
Timeout::~Timeout()
{
    alarm(0);
    signal(SIGALRM, SIG_DFL);
}

// Spawn a thread that runs the given function.
thread *spawn_thread(function<State()> func)
{
    thread *t = new thread(func);
    GLOBAL_TEMP_STATES_.threads.emplace_back(t);
    return t;
}

// Wait for all threads to finish.
void wait_all_threads()
{
    for (thread *t : GLOBAL_TEMP_STATES_.threads) {
        if (t->joinable()) {
            t->join();
        }
        delete t;
    }
    GLOBAL_TEMP_STATES_.threads.clear();
}

// Spawn a process that runs the given function.
int spawn_process(function<State()> func)
{
    pid_t pid = fork();
    if (pid < 0) {
        UNITTEST_UEXIT("fork() failed");
    } else if (pid == 0) {
        State ret = func();
        std::exit(ret);
    }
    GLOBAL_TEMP_STATES_.pids.push_back(pid);
    return (int)pid;
}

// Wait for all processes to finish.
void wait_all_processes()
{
    size_t nproc = GLOBAL_TEMP_STATES_.pids.size();
    for (size_t i = 0; i < nproc; ++i) {
        pid_t pid;
        int status;
        do {
            pid_t pid = wait(&status);
            if (pid == -1) {
                UNITTEST_UEXIT("wait() failed");
            }
        } while (!WIFEXITED(status));
        status = WEXITSTATUS(status);
        if (status != State::SUCCESS) {
            UNITTEST_EXIT((State)status, "process " + to_string(pid));
        }
    }
    GLOBAL_TEMP_STATES_.pids.clear();
}

// Run the given test function.
State test(function<State()> test_func)
{
    return test_func();
}

} // namespace unittest
} // namespace ark
