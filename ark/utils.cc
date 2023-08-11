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

/// Calculate the error rate between two values.
/// @tparam T Type of the values
/// @param a First value
/// @param b Second value
/// @return The error rate
template <typename T> float error_rate(T a, T b)
{
    T diff = abs(a - b);
    if (diff < numeric_limits<T>::min()) {
        return 0;
    }
    diff -= numeric_limits<T>::epsilon();
    T half_eps = numeric_limits<T>::epsilon() * T(0.5);
    if (a > b) {
        a -= half_eps;
        b += half_eps;
    } else {
        a += half_eps;
        b -= half_eps;
    }
    return (float)diff / max(abs((float)a), abs((float)b));
}

/// Calculate the error rate between two @ref half_t values.
/// @param a First value
/// @param b Second value
/// @return The error rate
float error_rate(half_t a, half_t b)
{
    return error_rate<half_t>(a, b);
}

/// Calculate the error rate between two floats.
/// @param a First value
/// @param b Second value
/// @return The error rate
float error_rate(float a, float b)
{
    return error_rate<float>(a, b);
}

/// Return mean squared error and max error rate between two matrices.
template <typename T>
pair<float, float> cmp_matrix(T *ground_truth, T *res, unsigned int m,
                              unsigned int n, unsigned int bs, unsigned int lm,
                              unsigned int ln, bool print)
{
    // TODO: deprecate this function.

    if (lm == 0) {
        lm = m;
    }
    if (ln == 0) {
        ln = n;
    }
    size_t num = (size_t)lm * (size_t)ln;

    const float thres_err = 0.01;

    float l2_loss = 0;
    float max_err = 0;
    // int cnt_flip = 0;
    // float max_err_gv;
    // float max_err_rv;
    for (unsigned int bidx = 0; bidx < bs; ++bidx) {
        for (unsigned int nidx = 0; nidx < n; ++nidx) {
            for (unsigned int midx = 0; midx < m; ++midx) {
                unsigned int idx = midx + nidx * lm + bidx * lm * ln;
                T gv = ground_truth[idx];
                T rv = res[idx];
                float diff = (float)(gv - rv);
                l2_loss += diff * diff;
                float err = error_rate(gv, rv);
                // if ((err > thres_err) && (error_rate(gv, -rv) <
                // thres_err) &&
                // (((float)gv * (float)rv) < 0)) {
                //     cnt_flip++;
                //     cout << (float)gv << "," << (float)rv << endl;
                //     cout << hex << gv.storage << "," << rv.storage << dec <<
                //     endl;
                // }
                if (err > max_err) {
                    max_err = err;
                    // max_err_gv = (float)gv;
                    // max_err_rv = (float)rv;
                }
            }
        }
    }
    if (print) {
        unsigned int x = 0;
        unsigned int cc = 0;
        cout << setprecision(4);
        for (unsigned int bidx = 0; bidx < bs; ++bidx) {
            for (unsigned int nidx = 0; nidx < n; ++nidx) {
                for (unsigned int midx = 0; midx < m; ++midx) {
                    unsigned int idx = midx + nidx * lm + bidx * lm * ln;
                    T exp = ground_truth[idx];
                    T act = res[idx];
                    if (error_rate(exp, act) < thres_err) {
                        cout << (float)act << ',';
                    } else {
                        cout << "\033[0;31m" << (float)act << "\033[0m,"
                             << "\033[0;32m" << (float)exp << "\033[0m,";
                    }
                    if (++cc == m) {
                        cout << '[' << x << ']' << endl;
                        cc = 0;
                        x++;
                    }
                }
            }
        }
    }
    // cout << max_err_gv << endl;
    // cout << max_err_rv << endl;
    // cout << cnt_flip << endl;
    return {l2_loss / num, max_err};
}

//
pair<float, float> cmp_matrix(half_t *ground_truth, half_t *res, unsigned int m,
                              unsigned int n, unsigned int bs, unsigned int lm,
                              unsigned int ln, bool print)
{
    // TODO: deprecate this function.

    return cmp_matrix<half_t>(ground_truth, res, m, n, bs, lm, ln, print);
}

//
pair<float, float> cmp_matrix(float *ground_truth, float *res, unsigned int m,
                              unsigned int n, unsigned int bs, unsigned int lm,
                              unsigned int ln, bool print)
{
    // TODO: deprecate this function.

    return cmp_matrix<float>(ground_truth, res, m, n, bs, lm, ln, print);
}

//
template <typename T>
void print_matrix(T *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln)
{
    // TODO: deprecate this function.

    unsigned int x = 0;
    unsigned int cc = 0;
    cout << setprecision(4);
    for (unsigned int bidx = 0; bidx < bs; ++bidx) {
        for (unsigned int nidx = 0; nidx < n; ++nidx) {
            for (unsigned int midx = 0; midx < m; ++midx) {
                unsigned int idx = midx + nidx * lm + bidx * lm * ln;
                T v = val[idx];
                cout << (float)v << ',';
                if (++cc == m) {
                    cout << '[' << x << ']' << endl;
                    cc = 0;
                    x++;
                }
            }
        }
    }
}

void print_matrix(half_t *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln)
{
    // TODO: deprecate this function.

    print_matrix<half_t>(val, m, n, bs, lm, ln);
}

void print_matrix(float *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln)
{
    // TODO: deprecate this function.

    print_matrix<float>(val, m, n, bs, lm, ln);
}

/// Return mean squared error and max error rate between two tensors.
/// @tparam T data type of the tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
template <typename T>
std::pair<float, float> tensor_compare(T *ground_truth, T *res, Dims shape,
                                       bool print = false)
{
    DimType nelem = shape.size();
    int ndims = shape.ndims();
    float l2_loss = 0;
    float max_err = 0;
    for (DimType i = 0; i < nelem; ++i) {
        float diff = (float)(ground_truth[i] - res[i]);
        l2_loss += diff * diff;

        float err = error_rate(ground_truth[i], res[i]);
        if (err > 0.) {
            if (print) {
                Dims idx;
                for (int j = 0; j < ndims; ++j) {
                    DimType vol = 1;
                    for (int k = j + 1; k < ndims; ++k) {
                        vol *= shape[k];
                    }
                    idx[j] = (i / vol) % shape[j];
                }
                std::cout << idx << " expected " << ground_truth[i]
                          << ", actually " << res[i] << " (err: " << err << ")"
                          << std::endl;
            }
            if (err > max_err) {
                max_err = err;
            }
        }
    }
    return {l2_loss / nelem, max_err};
}

/// Return mean squared error and max error rate between two @ref half_t
/// tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
std::pair<float, float> tensor_compare(half_t *ground_truth, half_t *res,
                                       Dims shape, bool print = false)
{
    return tensor_compare<half_t>(ground_truth, res, shape, print);
}

/// Return mean squared error and max error rate between two float tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
std::pair<float, float> tensor_compare(float *ground_truth, float *res,
                                       Dims shape, bool print = false)
{
    return tensor_compare<float>(ground_truth, res, shape, print);
}

/// Return mean squared error and max error rate between two int tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
std::pair<float, float> tensor_compare(int *ground_truth, int *res, Dims shape,
                                       bool print = false)
{
    return tensor_compare<int>(ground_truth, res, shape, print);
}

/// Return mean squared error and max error rate between two byte tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
std::pair<float, float> tensor_compare(uint8_t *ground_truth, uint8_t *res,
                                       Dims shape, bool print = false)
{
    return tensor_compare<uint8_t>(ground_truth, res, shape, print);
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
