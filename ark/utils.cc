// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iomanip>
#include <iostream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "include/ark.h"
#include "include/ark_utils.h"

// clang-format off
#include "vector_types.h"
#include "cutlass/half.h"
// clang-format on

using namespace std;

/// Convert cutlass::half_t to @ref ark::half_t
/// @param cuh cutlass::half_t
/// @return @ref ark::half_t
inline static const ark::half_t convert(const cutlass::half_t &cuh)
{
    ark::half_t ret;
    ret.storage = cuh.raw();
    return ret;
}

/// Numeric limits of @ref ark::half_t
template <> struct std::numeric_limits<ark::half_t>
{
    static ark::half_t max()
    {
        return convert(std::numeric_limits<cutlass::half_t>::max());
    }
    static ark::half_t min()
    {
        return convert(std::numeric_limits<cutlass::half_t>::min());
    }
    static ark::half_t epsilon()
    {
        return convert(std::numeric_limits<cutlass::half_t>::epsilon());
    }
};

ark::half_t operator+(ark::half_t const &lhs, ark::half_t const &rhs)
{
    return convert(cutlass::half_t::bitcast(lhs.storage) +
                   cutlass::half_t::bitcast(rhs.storage));
}

ark::half_t operator-(ark::half_t const &lhs, ark::half_t const &rhs)
{
    return convert(cutlass::half_t::bitcast(lhs.storage) -
                   cutlass::half_t::bitcast(rhs.storage));
}

ark::half_t operator*(ark::half_t const &lhs, ark::half_t const &rhs)
{
    return convert(cutlass::half_t::bitcast(lhs.storage) *
                   cutlass::half_t::bitcast(rhs.storage));
}

ark::half_t &operator+=(ark::half_t &lhs, ark::half_t const &rhs)
{
    cutlass::half_t v = cutlass::half_t::bitcast(lhs.storage) +
                        cutlass::half_t::bitcast(rhs.storage);
    lhs.storage = v.raw();
    return lhs;
}

ark::half_t &operator-=(ark::half_t &lhs, ark::half_t const &rhs)
{
    cutlass::half_t v = cutlass::half_t::bitcast(lhs.storage) -
                        cutlass::half_t::bitcast(rhs.storage);
    lhs.storage = v.raw();
    return lhs;
}

/// Return the absolute value of a @ref ark::half_t
/// @param val Input value
/// @return @ref Absolute value of `val`
ark::half_t abs(ark::half_t const &val)
{
    return convert(cutlass::abs(cutlass::half_t::bitcast(val.storage)));
}

namespace ark {

/// Construct a @ref half_t from a float
/// @param f Input value
half_t::half_t(float f)
{
    this->storage = cutlass::half_t(f).raw();
}

/// Convert a @ref half_t to a float
/// @return float
half_t::operator float() const
{
    return float(cutlass::half_t::bitcast(this->storage));
}

namespace utils {

/// Return a random @ref half_t array.
/// @param num Number of elements
/// @param max_val Maximum value
/// @return std::unique_ptr<half_t[]>
unique_ptr<half_t[]> rand_halfs(size_t num, float max_val)
{
    return rand_array<half_t>(num, max_val);
}

/// Return a random float array.
/// @param num Number of elements
/// @param max_val Maximum value
/// @return std::unique_ptr<float[]>
unique_ptr<float[]> rand_floats(size_t num, float max_val)
{
    return rand_array<float>(num, max_val);
}

/// Return a random bytes array.
/// @param num Number of elements
/// @return std::unique_ptr<uint8_t[]>
unique_ptr<uint8_t[]> rand_bytes(size_t num)
{
    return rand_array<uint8_t>(num, 255);
}

/// Return an array of values starting from `begin` with difference `diff`.
/// @tparam T Type of the array
/// @param num Number of elements
/// @param begin First value
/// @param diff Difference between two values
/// @return std::unique_ptr<T[]>
template <typename T>
unique_ptr<T[]> range_array(size_t num, float begin, float diff)
{
    T *ret = new T[num];
    for (size_t i = 0; i < num; ++i) {
        ret[i] = T(begin);
        begin += diff;
    }
    return unique_ptr<T[]>(ret);
}

/// Return a @ref half_t range array.
/// @param num Number of elements
/// @param begin First value
/// @param diff Difference between two values
/// @return std::unique_ptr<half_t[]>
unique_ptr<half_t[]> range_halfs(size_t num, float begin, float diff)
{
    return range_array<half_t>(num, begin, diff);
}

/// Return a float range array.
/// @param num Number of elements
/// @param begin First value
/// @param diff Difference between two values
/// @return std::unique_ptr<float[]>
unique_ptr<float[]> range_floats(size_t num, float begin, float diff)
{
    return range_array<float>(num, begin, diff);
}

/// Spawn a process that runs `func`.
/// @param func function to run in the spawned process.
/// @return PID of the spawned process.
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

/// Wait for a spawned process with PID `pid`.
/// @param pid PID of the spawned process.
/// @return -1 on any unexpected failure, otherwise return the exit status.
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

/// Wait for multiple child processes.
/// @param pids PIDs of the spawned processes.
/// @return 0 on success, -1 on any unexpected failure, otherwise the first seen
/// non-zero exit status.
int proc_wait(const vector<int> &pids)
{
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

} // namespace utils
} // namespace ark
